import modal
from pydantic import BaseModel

stub = modal.Stub("terrarium-dataset-upload")

# MODELS
class File(BaseModel):
    file_name: str
    base_file_name: str
    dataset_id: int
    user_id: str
    header_mapping: str

# IMAGES
process_new_file_image = modal.Image.debian_slim().pip_install("pandas")
get_data_image = modal.Image.debian_slim().pip_install("pandas", "google-cloud-storage")
process_df_image = modal.Image.debian_slim().pip_install("pandas")
explode_and_reorder_image = modal.Image.debian_slim().pip_install("pandas")
generate_embeddings_image = modal.Image.debian_slim().pip_install("pandas")
upload_to_pinecone_image = modal.Image.debian_slim().pip_install("pinecone-client", "pandas")
upload_to_prisma_image = modal.Image.debian_slim().pip_install("prisma", "pandas")

# UTILITY IMAGES
contact_websockets_image = modal.Image.debian_slim().pip_install("websockets")
update_dataset_image = modal.Image.debian_slim().pip_install("prisma")
clean_and_filter_message_image = modal.Image.debian_slim().pip_install("openai", "pandas")
get_embedding_threaded_image = modal.Image.debian_slim().pip_install("openai", "pandas")

# FUNCTIONS
@stub.function(image=process_new_file_image, timeout=2400)
@modal.web_endpoint(method="POST", wait_for_response=False)
def process_new_file(req_json: File, debug=True):
    import pandas as pd
    import json


    print("REQ_JSON:", req_json)

    file_name = req_json.file_name
    base_file_name = req_json.base_file_name

    dataset_id = file_name[:file_name.find("-")]
    database_dataset_id = req_json.dataset_id

    print('HEADER MAPPING:', json.loads(req_json.header_mapping))

    print("FILE_NAME:", file_name)
    try:
        contact_websockets.call("server_contacted", "complete", file_name)
        update_dataset.call(database_dataset_id, 'server_contacted')
        mapping_dict = json.loads(req_json.header_mapping)
        df = get_data.call(base_file_name, mapping_dict)

        print('INITIAL DF', df.head())

        df = df[:10] if debug else df
        
        updated_df = process_df.call(df)
    
        print("UPDATED DF:", updated_df.head())

        exploded_df = explode_and_reorder.call(updated_df, dataset_id)

        update_dataset.call(database_dataset_id, 'frs_generated', num_frs=len(exploded_df))
        contact_websockets.call("frs_generated", len(exploded_df), file_name)

        vector_df = generate_embeddings.call(exploded_df)

        print("VECTOR DF:", vector_df.head())
       
        update_dataset.call(database_dataset_id, 'vectors_generated')
        contact_websockets.call("vectors_generated", "complete", file_name)

        upload_to_pinecone.call(vector_df)
        upload_to_prisma.call(vector_df, req_json.user_id, req_json.dataset_id)

        update_dataset.call(database_dataset_id, 'dataset_generated')
        contact_websockets.call("dataset_generated", "complete", file_name)

    except RuntimeError as err:
        contact_websockets.call("known_error", str(err), file_name)
        update_dataset.call(database_dataset_id, 'known_error')

    except Exception as err:
        contact_websockets.call("unknown_error", str(err), file_name)
        update_dataset.call(database_dataset_id, 'unknown_error')

    return "Process Terminating"

@stub.function(mounts=[modal.Mount.from_local_file("./credentials.json", remote_path="/root/credentials.json")], image=get_data_image, timeout=300)
def get_data(blob, mapping_dict, bucket='terrarium-fr-datasets'):
    import os
    import io
    import pandas as pd
    from google.cloud import storage

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  = './credentials.json'

    try:
        client = storage.Client()
        terrarium_bucket = client.get_bucket(bucket)
        terrarium_blob = terrarium_bucket.get_blob(blob)
        blob_string = terrarium_blob.download_as_string(client=None)
        df = pd.read_csv(io.BytesIO(blob_string), on_bad_lines='warn', engine='python')

        # map data using provided dictionary
        print("MAPPING DATA:", mapping_dict)
        inverse_mapping_dict = {v: k for k, v in mapping_dict.items()}

        df = df.rename(columns=inverse_mapping_dict)
    
        print("DF COLUMNS:", df.columns)

        # remove everything that's not relevant
        return df[['text', 'author', 'id', 'created_at']]
    except Exception as err:
        raise RuntimeError("Get Data Error") from err



@stub.function(image=process_df_image, secret=modal.Secret.from_name("terrarium-secrets"), timeout=1200)
def process_df(df):
    import pandas as pd
    try:
        processing_output = [result for result in clean_and_filter_message.map(df['text'].to_list())]

        label, feature_requests = zip(*processing_output)

        df.loc[:, 'label'], df.loc[:, 'feature_requests'] = label, feature_requests

        updated_df = df[df.label == 1]
        return updated_df
    
    except Exception as err:
        raise RuntimeError("Process DF Error") from err

@stub.function(image=explode_and_reorder_image, timeout=600)
def explode_and_reorder(df, dataset_id):
    import pandas as pd
    
    try:
        exploded_df = df.explode('feature_requests')
        exploded_df = exploded_df.reset_index(drop=True)

        # renaming and generating ids
        exploded_df.rename(columns={"id": "message_id", "feature_requests": "feature_request"}, inplace=True)
        exploded_df["fr_id"] = exploded_df.apply(lambda x: f"{dataset_id}{x['message_id']}{hash(x['feature_request'])}", axis=1)

        # deals with cases where there's the same message and feature request text (i.e. GPT error)
        exploded_df = exploded_df.drop_duplicates(subset=["fr_id"])

        return exploded_df
    except Exception as err:
        raise RuntimeError("Get Data Error") from err

@stub.function(image=generate_embeddings_image, timeout=1200)
def generate_embeddings(df):
    import pandas as pd
    try:
        embeddings = [embedding for embedding in get_embedding_threaded.map(df['text'].to_list())]
        df["sim_embedding"] = embeddings
        return df
    except Exception as err:
        raise RuntimeError("Get Data Error") from err

@stub.function(image=upload_to_pinecone_image, secret=modal.Secret.from_name("terrarium-secrets"))
def upload_to_pinecone(df, timeout=600):
    import os
    import pinecone
    import itertools

    def preprocess_embeddings(dataframe):
        embeddings = df.loc[:, ['fr_id', 'sim_embedding']]
        embeddings.columns = ['id', 'values']
        return embeddings
    
    def batches(iterable, batch_size):
        """Helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = list(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = list(itertools.islice(it, batch_size))

    def batch_upload(vector_list, pinecone_index, batch_size):
        for vector_batch in batches(vector_list, batch_size=batch_size):
            pinecone_index.upsert(vectors=vector_batch)
    
    def async_batch_upload(vector_list, batch_size=100):
        with pinecone.Index("terrarium", pool_threads=30) as index:
            async_results = [
                index.upsert(vectors=id_vectors_chunk, async_req=True)
                for id_vectors_chunk in batches(vector_list, batch_size)
            ]

            return [async_result.get() for async_result in async_results]
    
    def schematize_and_upload_embeddings(unprocessed_data):
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp")
        index = pinecone.Index("terrarium")

        data = preprocess_embeddings(unprocessed_data)
        data_itertuples = list(data.itertuples(index=False, name=None))
        return async_batch_upload(data_itertuples)

    try: schematize_and_upload_embeddings(df)
    except Exception as err: raise RuntimeError("Pinecone Upload Error") from err

@stub.function(mounts=[modal.Mount.from_local_dir("./prisma", remote_path="/root/prisma")],
               secret=modal.Secret.from_name("terrarium-secrets"), 
               image=upload_to_prisma_image,
               timeout=600)
async def upload_to_prisma(df, user_id, dataset_id):
    import subprocess

    subprocess.run('prisma generate', shell=True)

    from prisma import Prisma

    try:
        db = Prisma()
        await db.connect()
        batcher = db.batch_()

        output_dicts = df.to_dict(orient='records')

        for fr in output_dicts:
            batcher.featurerequest.upsert(
                where={
                    "fr_id": fr["fr_id"]
                },
                data = {
                'create': {
                    "fr_id": fr["fr_id"],
                    "message_id": str(fr["message_id"]),
                    "message": fr["text"],
                    "created_at": fr['created_at'],
                    "author": fr["author"],
                    "fr": fr['feature_request'],
                    "userId": user_id,
                    "datasetId": dataset_id,
                    },
                'update': {
                    "fr_id": fr["fr_id"],
                    "message_id": str(fr["message_id"]),
                    "message": fr["text"],
                    "created_at": fr['created_at'],
                    "author": fr["author"],
                    "fr": fr['feature_request'],
                    "userId": user_id,
                    "datasetId": dataset_id,
                    }   
                }
            )
        
        await batcher.commit()

        await db.disconnect()
    except Exception as err:
        raise RuntimeError("Upload to Prisma Error") from err


# UTILITIES
@stub.function(image=contact_websockets_image,
               secret=modal.Secret.from_name("terrarium-secrets"),
               timeout=1800)
async def contact_websockets(type, status, dataset):
    import websockets
    import json
    import os
    try:
        async with websockets.connect(os.environ["TERRARIUM_WEBSOCKET"]) as ws:
            await ws.send(json.dumps({"type": type, "status": status, "dataset": dataset, "pipeline": "dataset"}))

            msg = await ws.recv()
            print("Message:", msg)

            await ws.close()

            return

    except Exception as err:
        print("ERROR:", err)
        raise RuntimeError("Contact Websockets Error") from err



@stub.function(mounts=[modal.Mount.from_local_dir("./prisma", remote_path="/root/prisma")], image=update_dataset_image)
async def update_dataset(database_dataset_id, status, num_frs=None):
    import subprocess

    subprocess.run('prisma generate', shell=True)

    from prisma import Prisma

    try:
        db = Prisma()
        await db.connect()

        if num_frs:
            print("NUM FRS:", num_frs)
            await db.dataset.update(
                where={
                    "datasetId": database_dataset_id
                },
                data={
                    "status": status,
                    "size": str(num_frs)
                }
            )
        else:
            await db.dataset.update(
                where={
                    "datasetId": database_dataset_id
                },
                data={
                    "status": status
                }
            )
        await db.disconnect()
    
    except Exception as err:
        raise RuntimeError("Update Dataset Error") from err


@stub.function(secret=modal.Secret.from_name("terrarium-secrets"), image=clean_and_filter_message_image, concurrency_limit=100)
def clean_and_filter_message(message):
    import os
    import openai
    import json
    openai.api_key = os.environ["OPENAI_KEY"]

    def filter_and_classify(message, top_p=0.15):
            prompt = f"""
                If the message below contains at least one feature request, respond "Yes:" and then list all features that the user is requesting in a
                Python list and do not provide any other commentary. Make each feature request in the list a full sentence. Use imperative tense and be specific.
                If the message does not contain a feature request, do not print anything except for the word "No".

                Message: {message}
            """
            res = openai.Completion.create(model="text-davinci-003", 
                                                prompt=prompt,
                                                top_p=top_p, 
                                                max_tokens=200)
            return res['choices'][0]['text']

    try:    
        text = filter_and_classify(message).strip()
        if text.lower() == 'no': return (0, "N/A")
        

        sep = text.find(":")
        if sep == -1: return (-1, "N/A")
        elif text[:sep].lower() != "yes": return (-2, "N/A")
        else: 
            try:
                lst = json.loads(text[sep+1:].strip())
                return (1, lst)
            except: return (-3, "N/A")
    except Exception as err:
        raise RuntimeError("Clean and Filter Error") from err


@stub.function(secret=modal.Secret.from_name("terrarium-secrets"), image=get_embedding_threaded_image, concurrency_limit=100)
def get_embedding_threaded(message):
    import openai
    import os

    try:
        openai.api_key = os.environ["OPENAI_KEY"]

        message_embedding = openai.Embedding.create(input=[message], model="text-embedding-ada-002")['data'][0]['embedding']
        
        return message_embedding
    except Exception as err:
        raise RuntimeError("Get Embedding Threaded Error") from err












####### TESTING ####### 
# process_df_image = modal.Image.debian_slim().pip_install("pandas")
# @stub.function(image=process_df_image)
# def test_process_df():
#     import pandas as pd
#     import time

#     start = time.time()
#     df_dict = {'text': {0: '!history',
#                 1: "thanks for the feedback! i'll discuss it with the team",
#                 2: 'a couple feature requests:\n\n- it would be fantastic if whiteboards could be collapsed / folded to show only their title, the same way cards can!\n\n- it would be nice to be able to set the size of a collapsed card\'s title.\n-- often i only want the title of a card showing, but for readability i wish i could set the text size to h1 or h2 size, for example. the collapsed card\'s text is very small for me currently. i get around this by using the card unfolded and resizing the card to hide the rest of the text, but i have to add a block with a period to get an even margin of whitespace around the title (see screenshot - on the 3rd card, the first line of body text is visible and distracting, compared to the other two where i "fake" a full margin and all attention goes to the title).',
#                 3: "yep, when letting go of the connection arrow in an empty space, a new, connected card is created!\n\nright now there's unnecessary friction. i have to first create a new card and then select the connection tool, and then connect the cards.\n\ndidn't catch the previous request. liked it too",
#                 4: "if i may clarify, you want to create a new card *when dragging the connection tool (arrow) onto an empty whiteboard space*? (we can already create a new card by double clicking in a whiteboard space).\nthis has been suggested (with 10 likes) not too long ago. it doesn't hurt to request again, though!\nhttps://discord.com/channels/812292969183969301/856016076311101470/970595454645575760",
#                 5: 'create a new card by clicking in empty space in whiteboard when using the whiteboard app',
#                 6: 'yes i want calendar system too and if can link with google calendar will be cool',
#                 7: 'october',
#                 8: 'when (which month) can we expect the web version to come out?',
#                 9: "when clicking outside of a card on a whiteboard, it would be nice if any selected text could be automatically de-selected. once i click outside the card, i'm done working in it for the moment, and am reviewing the whole picture. it's visually bothering to have the selection still visible on the card, and i have to click into the card, deselect, and click back out which disrupts my flow."},
#                 'author': {0: 'finn',
#                 1: 'Alan Chan',
#                 2: 'robotic_scarab',
#                 3: 'maxlinworm',
#                 4: 'Sams_Here',
#                 5: 'maxlinworm',
#                 6: 'aran',
#                 7: 'Alan Chan',
#                 8: 'duolidoris',
#                 9: 'robotic_scarab'},
#                 'id': {0: '1019191342687072276',
#                 1: '1019075643096432781',
#                 2: '1019062030176616458',
#                 3: '1019003148720951346',
#                 4: '1018986357592445019',
#                 5: '1018982692655726672',
#                 6: '1018254575049916507',
#                 7: '1016887622272041021',
#                 8: '1016881351049367632',
#                 9: '1016745192277159936'},
#                 'created_at': {0: '2022-09-13 10:22:31.756000+00:00',
#                 1: '2022-09-13 02:42:46.824000+00:00',
#                 2: '2022-09-13 01:48:41.251000+00:00',
#                 3: '2022-09-12 21:54:42.818000+00:00',
#                 4: '2022-09-12 20:47:59.501000+00:00',
#                 5: '2022-09-12 20:33:25.712000+00:00',
#                 6: '2022-09-10 20:20:08.947000+00:00',
#                 7: '2022-09-07 01:48:22.008000+00:00',
#                 8: '2022-09-07 01:23:26.832000+00:00',
#                 9: '2022-09-06 16:22:24.051000+00:00'}}
#     df = pd.DataFrame.from_dict(df_dict)
#     result = process_df(df)

#     print("RESULT:", result)
#     end = time.time()

#     print("TIMING:", end - start)

#     return end - start
