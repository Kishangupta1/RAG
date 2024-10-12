from flask import Flask, request
from flask.views import MethodView
from flask_smorest import Api, Blueprint, fields as smo_fields
from flask_cors import CORS
import uuid
from marshmallow import Schema, fields
from marshmallow.utils import _Missing
import json
import time
import os
import tempfile
import hashlib
import logging

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.runnables import RunnableLambda
from chromadb.config import Settings
from langchain_core.runnables import ConfigurableField
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyMuPDFLoader, DirectoryLoader, CSVLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader,
    UnstructuredHTMLLoader, UnstructuredPowerPointLoader, UnstructuredMarkdownLoader, JSONLoader, UnstructuredURLLoader)

logging.basicConfig(level=logging.DEBUG)

file_type_mappings = {
    '*.txt': TextLoader,
    '*.pdf': PyMuPDFLoader,
    '*.csv': CSVLoader,
    '*.docx': Docx2txtLoader,
    '*.xlss': UnstructuredExcelLoader,
    '*.xlsx': UnstructuredExcelLoader,
    '*.html': UnstructuredHTMLLoader,
    '*.pptx': UnstructuredPowerPointLoader,
    '*.ppt': UnstructuredPowerPointLoader,
    '*.md': UnstructuredMarkdownLoader,
    '*.json': JSONLoader,
}


class Config:
    API_TITLE = '''Parabole TRAIN LLM APIs'''
    API_VERSION = 'v1'
    OPENAPI_VERSION = '3.0.3'
    OPENAPI_URL_PREFIX = '/'
    OPENAPI_SWAGGER_UI_PATH = '/'
    OPENAPI_SWAGGER_UI_URL = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.3/'
    OPENAPI_SWAGGER_UI_CONFIG = {'docExpansion': 'none'}  # 'supportedSubmitMethods': []
    OPENAPI_RAPIDOC_PATH = '/rapidoc'
    OPENAPI_RAPIDOC_URL = 'https://cdn.jsdelivr.net/npm/rapidoc/dist/rapidoc-min.js'
    OPENAPI_RAPIDOC_CONFIG = {'show-method-in-nav-bar': 'as-colored-block', 'use-path-in-nav-bar': 'true',
                              'render-style': 'focused', 'allow-server-selection': 'false',
                              'show-header': 'false'}
    OPENAPI_REDOC_PATH = "/redoc"
    OPENAPI_REDOC_URL = "https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"


app = Flask(__name__)
CORS(app)
app.config.from_object(Config)
api = Api(app, spec_kwargs={'info': {"description": '''These APIs provide a gateway to use the Local LLM installed.'''}})
bp_llm = Blueprint('LLM Utility', 'Random', url_prefix='/api/train/llm')

llm = ChatOllama(model="llama3:70b-instruct")  # :70b-instruct
llm_embeddings = HuggingFaceEmbeddings(model_name="llmrails/ember-v1")
vector_store = Chroma(collection_name='parabole',
                      embedding_function=llm_embeddings,
                      persist_directory='./parabole_vectorstore',
                      client_settings= Settings(anonymized_telemetry=False),
                      collection_metadata={"hnsw:space": "cosine"})
history = {}
similarity_prompt = {
    'system': 'You are tasked with finding the most similar texts (from an existing set of paragraphs) for the given '
              'input text. Your response must be based on semantic similarity. '
              'Whenever you find a match, try to assign a "semantic similarity score", a decimal number between 0.0 and'
              ' 1.000, where 0.000 is least semantically similar and 1.0 is a perfect semantic match. '
              'Provide up to {limit} most similar results. '
              'The format of the result must be: {{<paragraph number>:<score>,...}}.',
    'user': 'Use the following existing set of paragraphs for your analysis.'
            ' Each paragraph is prefixed with <paragraph line number.>:\n'
            '{vectorstore_results}\n\n'
            'Your input text is:\n'
            '{query_text}\n'
}


def main():
    return app


def save_in_temp_dir(files):
    temp_dir = tempfile.mkdtemp()
    for file in files:
        with open(os.path.join(temp_dir, file.filename), 'wb') as f:
            f.write(file.read())
    return temp_dir


def inspect(state):
    print(state)
    return state


class RAGInput(Schema):
    class Message(Schema):
        role = fields.String(load_default='user')
        content = fields.String(load_default='')

    class Configuration(Schema):
        mirostat_tau = fields.Float(load_default=1.0)
        num_ctx = fields.Int(load_default=2048)
        num_predict = fields.Int(load_default=-1)
        repeat_last_n = fields.Int(load_default=64)
        repeat_penalty = fields.Float(load_default=1.1)
        stop = fields.List(fields.String(load_default=''))
        temperature = fields.Float(load_default=0.0)
        top_k = fields.Int(load_default=1)
        top_p = fields.Float(load_default=0.9)

    configuration = fields.Nested(Configuration())
    messages = fields.List(fields.Nested(Message()))
    context_file = fields.List(smo_fields.Upload())
    context_link = fields.List(fields.String())
    history_prev_msg_id = fields.String(load_default='', )
    query = fields.String(load_default='')


class RAGOutput(Schema):
    msg_id = fields.String(load_default='', )
    content = fields.String(load_default='', )


class EmbeddingInput(Schema):
    namespace = fields.String(load_default='')
    texts = fields.List(fields.String(load_default=''))
    text_file = smo_fields.Upload()


class EmbeddingSearch(Schema):
    namespace = fields.String(load_default='')
    texts = fields.List(fields.String(load_default=''))
    text_file = smo_fields.Upload()
    limit = fields.Int(load_default=5)


class EmbeddingOutput(Schema):
    class Result(Schema):
        score = fields.Float()
        text = fields.String()

    input_text = fields.String()
    results = fields.List(fields.Nested(Result()))


class EmbeddingDelete(Schema):
    namespace = fields.String(load_default='')
    texts = fields.List(fields.String(load_default=''))


@bp_llm.route('/embeddings')
class Embeddings(MethodView):
    @bp_llm.arguments(EmbeddingInput, location='files')
    def post(self, inputs):
        """ Create vector embeddings for a given set of sentences. It 'namespace' is reused, the embeddings would
            append to the existing set.
        """
        inputs = request.form
        nsp = inputs.get('namespace')
        texts = inputs.getlist('texts')
        texts = [sub_item.replace('\\n', '') for item in texts for sub_item in item.split('\\n,') if sub_item]
        if file := request.files.get('text_file'):
            texts.extend(str(file.read(), encoding='utf-8').split('\r\n'))

        metadata = [{nsp: True} if nsp else {} for _ in range(len(texts))]
        ids = [hashlib.md5(bytes(text, 'utf-8')).hexdigest() for text in texts]
        collection = vector_store.get(ids, include=["metadatas"])
        old_ids = collection['ids']
        old_metadata = collection['metadatas']

        create_ids = []
        update_ids = []
        for id_, metadata_, text in zip(ids, metadata, texts):
            if nsp:
                if id_ in old_ids:
                    old_metadata_ = old_metadata[old_ids.index(id_)]
                    if not old_metadata_:
                        old_metadata_ = {}
                    if not old_metadata_.get(nsp, False):
                        old_metadata_[nsp] = True
                        update_ids.append((text, old_metadata_, id_))
                else:
                    create_ids.append((text, metadata_, id_))
            else:
                if id_ not in old_ids:
                    create_ids.append((text, metadata_, id_))

        if update_ids:
            ids = [id_ for _, _, id_ in update_ids]
            docs = [Document(page_content=text, metadata=metadata) for text, metadata, _ in update_ids]
            vector_store.update_documents(ids, docs)
        if create_ids:
            texts, metadata, ids = map(list, zip(*create_ids))
            vector_store.add_texts(texts=texts, metadatas=metadata, ids=ids)
            '''texts = [text for text, _, _ in update_ids]
            metadata = [metadata_ for _, metadata_, _ in update_ids]
            ids = [id_ for _, _, id_ in update_ids]
            vector_store.add_documents() update_documents(ids, docs)'''
        print("update_ids", update_ids)
        print("create_ids", create_ids)
        return 'Sentences embedded successfully!'

    @bp_llm.arguments(EmbeddingDelete, location='json')
    def delete(self, inputs):
        """
        Deletes a vector 'embedding' text or all vector 'embeddings' present in a namespace
        :param inputs:
        :return:
        """

        # print(vector_store.get())
        nsp = inputs.get('namespace')
        texts = inputs.get('texts')
        texts = [sub_item.replace('\\n', '') for item in texts for sub_item in item.split('\\n,') if sub_item]

        ids = [hashlib.md5(bytes(text, 'utf-8')).hexdigest() for text in texts]
        collection = vector_store.get(ids, where={nsp: True} if nsp else None)
        ids = collection['ids']
        metadata = collection['metadatas']
        texts = collection['documents']
        delete_ids = []
        update_ids = []
        if not nsp:
            delete_ids = ids
        else:
            for id_, metadata_, text in zip(ids, metadata, texts):
                metadata_[nsp] = False
                if any(metadata_.values()):
                    update_ids.append((text, metadata_, id_))
                else:
                    delete_ids.append(id_)
        if update_ids:
            ids = [id_ for _, _, id_ in update_ids]
            docs = [Document(page_content=text, metadata=metadata) for text, metadata, _ in update_ids]
            vector_store.update_documents(ids, docs)
        if delete_ids:
            vector_store.delete(delete_ids)
        return 'Embeddings deleted successfully.'


@bp_llm.route('/embeddings/match')
class Embeddings(MethodView):
    @bp_llm.arguments(EmbeddingSearch, location='files')
    @bp_llm.response(200, EmbeddingOutput(many=True))
    def post(self, inputs):
        """ Performs a vector similarity match between the given set of sentences and the vectors in
            the given namespace.
        """
        def process_through_llm(text):
            def format_llm_output():
                # logging.debug(llm_result)
                llm_result_ = '{}'
                for line in llm_result.splitlines():
                    llm_result_ = line.strip()
                    if llm_result_.startswith('{'):  # if { found, eureka!!!
                        break
                    if llm_result_[:1].isdigit():  # if digit found, wrap the line as {line}
                        llm_result_ = f'{{{llm_result_}}}'
                        break
                    llm_result_ = '{}'   # no matching result on this line, try looking into the next line

                result_dict = {}
                try:
                    result_dict = eval(llm_result_)  # if it still gives any issues, ignore everything!
                except: pass
                logging.debug(result_dict)
                return [(Document(page_content=results[int(k)-1]), 1.0-float(v)) for k, v in result_dict.items() if k]

            if not docs:
                return []

            results = []
            result_texts = []
            for i, (doc, score) in enumerate(docs):
                results.append(doc.page_content)
                result_texts.append(f'{i + 1}. {doc.page_content}')
                logging.debug(f'{i+1}, {score}, {doc.page_content}')

            # logging.debug('\n'.join(result_texts))
            messages = []
            messages.append(SystemMessage(content=similarity_prompt['system'].format(limit=limit)))
            messages.append(HumanMessage(content=similarity_prompt['user']
                                         .format(vectorstore_results='\n'.join(result_texts), query_text=text)))
            # logging.debug(messages)
            st = time.time()
            llm_result = (ChatPromptTemplate.from_messages(messages) | llm | StrOutputParser()).invoke({})
            et = time.time() - st
            logging.debug(f'time taken to search llm: {et} sec')
            return format_llm_output()

        def process_inputs():
            inputs = request.form
            texts_ = inputs.getlist('texts')
            texts_ = [sub_item.replace('\\n', '') for item in texts_ for sub_item in item.split('\\n,') if sub_item]
            if file := request.files.get('text_file'):
                texts_.extend(str(file.read(), encoding='utf-8').split('\r\n'))
            nsp_ = inputs.get('namespace')
            limit_ = int(limit_) if (limit_ := inputs.get('limit')) else 5
            return texts_, nsp_, limit_

        texts, nsp, limit = process_inputs()
        output = []
        filter_ = {nsp: True} if nsp else None
        for text, vec in zip(texts, llm_embeddings.embed_documents(texts)):
            # logging.debug(f'filter: {filter_}')
            st = time.time()
            docs = vector_store.similarity_search_by_vector_with_relevance_scores(vec, int(limit * 2), filter=filter_)
            et = time.time() - st
            logging.debug(f'time taken to search vectorstore: {et} sec')
            # logging.debug(docs)
            # docs = process_through_llm(text)
            # logging.debug(docs)
            res = [EmbeddingOutput.Result().load({'text': doc.page_content, 'score': round(1-score, 5)})
                   for doc, score in docs]
            output.append(EmbeddingOutput().load({'input_text': text, 'results': res[:limit]}))
        return output


@bp_llm.route('/rag/invoke', methods=['POST'])
@bp_llm.arguments(RAGInput, location='files')
@bp_llm.response(200, RAGOutput)
def invoke(inputs):
    """ Performs a RAG operation given set of messages, contexts and query prompt. If one reuses any previously returned
        message ID in history, it includes everything from the previous conversations into the context.
    """

    def process_query():
        if query := inputs.get('query'):
            return [HumanMessage(content=query)]
        return []

    def process_messages():
        msgs = []
        if messages_ := inputs.getlist('messages'):
            try:
                messages_ = [json.loads(item) for item in messages_]  # take care of json.dumps array items
            except ValueError as e:
                messages_ = json.loads(f'[{messages_[0]}]')  # take care of weird Swagger UI array items
            for message in messages_:
                if type(message) is dict:
                    role = message.get('role')
                    content = message.get('content')
                    if not role or role == 'user':
                        msgs.append(HumanMessage(content=content))
                    elif role == 'system':
                        msgs.append(SystemMessage(content=content))
                    elif role == 'assistant':
                        msgs.append(AIMessage(content=content))
        return msgs

    def get_history():
        return history.get(msg_id, {}).get('messages', [])

    def set_history():
        def clean_older_entries():
            current_time = time.time()
            for msg in list(history.keys()):
                if history[msg_id]['time'] + 60 * 60 * 1 < current_time:
                    del history[msg]

        history[msg_id] = history.get('msg_id', {'time': None, 'messages': []})
        history[msg_id]['messages'].extend(messages)
        history[msg_id]['messages'].extend([AIMessage(content=result)])
        history[msg_id]['time'] = time.time()
        clean_older_entries()

    def process_context():
        def process_files():
            if files := request.files.getlist('context_file'):
                temp_dir = save_in_temp_dir(files)
                for glob_pattern, loader_cls in file_type_mappings.items():
                    try:
                        if loader_cls == JSONLoader:
                            kw = {'jq_schema': '.', 'text_content': False}
                        elif loader_cls == TextLoader:
                            kw = {'autodetect_encoding': True}
                        else:
                            kw = None
                        loader_dir = DirectoryLoader(temp_dir, glob=glob_pattern, loader_cls=loader_cls,
                                                     loader_kwargs=kw)
                        docs.extend(loader_dir.load_and_split())
                    except Exception as e:
                        print(e)
                        continue

        def process_links():
            if links := inputs.getlist('context_link'):
                links = [sub_item for item in links for sub_item in item.split(',') if sub_item]
                url_loader = UnstructuredURLLoader(urls=links)
                docs.extend(url_loader.load_and_split())

        docs = []
        process_files()
        process_links()

        if docs:
            vs = FAISS.from_documents(documents=docs, embedding=llm_embeddings)
            docs = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3}).invoke(inputs['query'])
            content = '\n'.join([doc.page_content for doc in docs])
            return [SystemMessage(content=content)]
        return []

    def set_configuration1(llm_):
        if config := inputs.get('configuration'):
            for k, v in json.loads(config).items():
                if type(v) is list:
                    v = [item for item in v if item]
                setattr(llm_, k, v)
        print(llm)
        logging.debug(llm)
        return llm_

    def set_configuration(llm_):
        if config := inputs.get('configuration'):
            config = json.loads(config)
            for k, v in config.items():
                if type(v) is list:
                    config[k] = [item for item in v if item]
            llm_ = llm.with_config(configurable=config)
        return llm_

    inputs = request.form
    if not (msg_id := inputs.get('history_prev_msg_id')):
        msg_id = str(uuid.uuid4())

    messages = []
    messages += process_messages()
    messages += get_history()
    messages += process_context()
    messages += process_query()
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | set_configuration(llm) | StrOutputParser()
    # chain = prompt | RunnableLambda(inspect) | llm | StrOutputParser()
    result = chain.invoke({})

    set_history()
    return RAGOutput().load({'msg_id': msg_id, 'content': result})


def configure_llm():
    def configure_defaults(llm_):
        for fld, value in RAGInput.Configuration._declared_fields.items():
            if type(value.load_default) is not _Missing:
                setattr(llm_, fld, value.load_default)

    def configure_overrides(llm_):
        return llm_.configurable_fields(**{fld: ConfigurableField(id=fld)
                                           for fld in RAGInput.Configuration._declared_fields})

    global llm
    configure_defaults(llm)
    llm = configure_overrides(llm)


configure_llm()
api.register_blueprint(bp_llm)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
