import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from gtts import gTTS # need to install
import time
import random
from pypdf import PdfReader
import base64
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#####     CONSTANTS     #####
OPENAI_APIKEY = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = 'text-embedding-3-large'  # 'text-embedding-3-small'
#SPACY_MODEL = spacy.load('en_core_web_sm') # 'en_core_web_lg'
APP_NAME = "Government ProcessEase"
APP_DESC = "GovEase: Making government processes a breeze"
lang = "English"
speak = False
volume = 0.5
COLOR_BLUE = "#0c2e86"
COLOR_RED = "#a73c07"
COLOR_YELLOW = "#ffcd34"
COLOR_GRAY = '#f8f8f8'
# Suggested questions based on selected process
suggested_questions = {
    "LTO Licensing Process": [
        "Can I drive with an expired driver’s license?",
        "Do I need to retake the driving test when renewing a driver’s license?",
        "Can a driver’s license be renewed after it expires?",
        "Can I still renew my driver’s license if it has expired for over 10 years?",
        "Can I skip on the Comprehensive Driver’s Education (CDE)?",
        "What is the age limit for driver’s license renewal in the Philippines?",
        "Is it possible to renew my license on weekends?",
        "What documents are needed for renewing a driver’s license?",
        "Is there a penalty fee for late renewal of a driver’s license?",
        "Is there a grace period for renewing an expired driver's license?"
    ],
    "Voter Registration Process": [
        "How do I register to vote?",
        "What are the eligibility requirements for voter registration?",
        "Where can I register to vote?",
        "What documents do I need to bring for voter registration?",
        "Is there a deadline for voter registration?",
        "Can I check my voter registration status online?",
        "What are the steps involved in the voter registration process?",
        "Are there different registration processes for different types of elections (e.g., national, local)?",
        "Do I need to update my voter registration if I move to a different address?",
        "What should I do if I encounter issues or have questions during the voter registration process?"
    ],
    "PhilHealth Member Registration": [
        "What are the steps to apply for PhilHealth membership?",
        "Where can I apply for PhilHealth membership?",
        "What documents are required for PhilHealth membership application?",
        "Who is eligible to apply for PhilHealth membership?",
        "Is there an age limit for applying for PhilHealth membership?",
        "What are the benefits of becoming a PhilHealth member?",
        "Can I apply for PhilHealth membership online?",
        "How long does it take to process a PhilHealth membership application?",
        "How do I check the status of my PhilHealth membership application?",
        "What should I do if my PhilHealth membership application is denied?"
    ],
    "Taxpayer (TIN ID) Registration": [
        "What documents are required for TIN ID application?",
        "Can I authorize someone else to apply for my TIN ID?",
        "Where can I submit my TIN ID application?",
        "What are the steps to apply for a TIN ID?",
        "How long does it take to process a TIN ID application?",
        "What are acceptable IDs for TIN ID application?",
        "Is there any fee for applying for a TIN ID?",
        "Can I apply for a TIN ID online?",
        "Who is eligible to apply for a TIN ID?",
        "What should I do if my TIN ID is lost or damaged?"
    ],
    "National ID Application Process": [
        "What is the PhilSys National ID, and what benefits does it provide?",
        "Who is eligible to apply for the PhilSys National ID?",
        "What are the steps involved in the PhilSys National ID application process?",
        "What demographic information is required for the PhilSys application?",
        "What types of identification documents are acceptable for the PhilSys registration?",
        "Where can applicants secure the PhilSys Registration Form and other required documents?",
        "What happens if an applicant does not have any of the primary documents required for PhilSys registration?",
        "How is the uniqueness of an applicant's identity ensured during the PhilSys registration process?",
        "What biometric information is collected during the PhilSys registration?",
        "When can an applicant expect to receive their PhilSys National ID after registration?"
    ]
}
mp3_fp = BytesIO()

homepage_titles = {
    "English": {
        "problem": "The Problem",
        "prob_desc": "Understanding and navigating government processes is often difficult and time-consuming, leading to confusion and frustration. The lack of easily accessible, clear, and concise information contributes to inefficiency and lower civic engagement, especially for those with limited resources and those facing barriers such as language proficiency and visual impairments.",
        "sol": "The Solution",
        "sol_desc": "Our application simplifies government processes by providing a user-friendly platform where users can quickly query and retrieve relevant information in multiple languages.",
        "feat": "Features",
        "feat_titles": ["Interactive Chatbot", "Multilingual Support", "Text-to-Speech"],
        "feat_desc": ["Engage with a chatbot to get instant answers and guidance on various processes.", "Access information in your preferred language.", "Listen to information read aloud for easier comprehension."],
        "mission": "Our Mission",
        "mis_desc": "We empower citizens by simplifying government processes, promoting transparency, and fostering civic engagement. Our goal is to make essential services accessible and understandable for everyone. Through innovation and community collaboration, we support informed decision-making and active civic participation.",
        "start_title": "Getting Started",
        "query_title": "Query Page",
        "query_desc": "Easily search for information on various government processes, such as driver's license applications and voter registration. The page supports multiple languages and includes a text-to-speech feature for enhanced accessibility.",
        "query_how_to": f"1. Select the process you want to know more about\n2. Type your question in the 'Ask a question' bar located at the bottom of the screen\n3. Press the send button on the rightmost part or press the 'Enter' key of the keyboard\n4. Wait for the response of the app. This may take some time\n5. Ask as many questions as you want",
        "data_title": "Add to Database",
        "data_desc": "Administrators can contribute or update information on government processes. This ensures the database is current, comprehensive, and reflects the latest procedures, maintaining the app's inclusivity and utility.",
        "data_how_to": f"If you want to add a webpage to the database:\n\n1. Place the link in the text area\n2. Enter the title of the process\n3. Press the Upload! button\n4. Check the database to see if it has been successfully added.\n\nIf you want to add a PDF file:\n1. Click the 'Upload PDF'\n2. Upload the PDF file you wish to add\n3. Enter the title of the process\n4. Press the Upload! button\n\nNote: Make sure that you aren't uploading both a link and a PDF file simultaneously.",
        "how": "How to Use"
    },
    "Tagalog": {
        "problem": "Ang Problema",
        "prob_desc": "Ang pag-unawa at pag-naviga sa mga proseso ng pamahalaan ay madalas mahirap at nakakabawas ng oras, na nagdudulot ng kalituhan at pagkabalisa. Ang kakulangan ng madaling-access, malinaw, at maikliang impormasyon ay naglalagay ng kontribusyon sa hindi pagiging epektibo at mas mababang pakikilahok sa sibiko, lalo na para sa mga may limitadong mapagkukunan at yaong may mga hadlang tulad ng kakayahan sa wika at mga problema sa paningin.",
        "sol": "Ang Solusyon",
        "sol_desc": "Pinapadali ng aming aplikasyon ang mga proseso ng pamahalaan sa pamamagitan ng pagbibigay ng isang user-friendly na plataporma kung saan maaaring madali at mabilis na magtanong at kumuha ng kaugnay na impormasyon sa iba't ibang wika.",
        "feat": "Mga Tampok",
        "feat_titles": ["Interaktibong Chatbot", "Suporta sa Iba't Ibang Wika", "Text-to-Speech"],
        "feat_desc": ["Makipag-ugnayan sa isang chatbot upang makakuha ng agad-agad na mga sagot at patnubay sa iba't ibang mga proseso.", "Makakuha ng impormasyon sa iyong piniling wika.", "Pakinggan ang impormasyon na binabasa para mas madaling maintindihan."],
        "mission": "Ang Aming Misyon",
        "mis_desc": "Pinapalakas namin ang mga mamamayan sa pamamagitan ng pagpapadali ng mga proseso ng pamahalaan, pagsusulong ng transparency, at pagpapaigting ng partisipasyon sa sibika. Ang layunin namin ay gawing accessible at maliwanag para sa lahat ang mga impormasyon galing sa mga mahahalagang serbisyo ng gobyerno. Sa pamamagitan ng pagbabago at kooperasyon sa komunidad, tinutulungan namin ang maayos na paggawa ng desisyon at aktibong paglahok sa sibika.",
        "start_title": "Gabay sa Paggamit",
        "query_title": "Pahina Para sa mga Tanong",
        "query_desc": "Madaliang maghanap ng impormasyon tungkol sa iba't ibang proseso ng pamahalaan, tulad ng aplikasyon para sa lisensya ng driver at rehistrasyon ng botante. Ang pahina na ito ay sumusuporta sa iba't ibang wika at maaaring basahin nang malakas ang mga resulta (Text to Speech) para mas accessible ito.",
        "query_how_to": f"1. Pumili ng proseso na nais mong malaman ang higit pa tungkol dito.\n2. I-type ang iyong tanong sa bar na 'Magtanong' na nasa ibaba ng screen.\n3. Pindutin ang send button sa pinakakanang bahagi o pindutin ang 'Enter' key sa keyboard.\n4. Maghintay sa tugon ng aplikasyon. Maaaring tumagal ng ilang sandali ito.\n5. Magtanong ng kahit ilang tanong ang gusto mo.",
        "data_title": "Idagdag sa Database",
        "data_desc": "Maaaring idagdag o i-update ng mga tagapamahala ang impormasyon tungkol sa iba't ibang proseso. Ito ay tiyak na nagpapanatiling kasalukuyan, komprehensibo, at sumasalamin sa pinakabagong mga pamamaraan, na nagpapanatili sa kahalagahan at kapakinabangan ng aplikasyon.",
        "data_how_to": "Kung nais mong idagdag ang isang webpage sa database:\n\n1. Ilagay ang link sa text area\n2. Ipasok ang pamagat ng proseso\n3. Pindutin ang Upload! button\n4. Tingnan ang database upang makita kung ito ay matagumpay na idinagdag.\n\nKung nais mong idagdag ang isang PDF file:\n1. I-click ang 'Upload PDF'\n2. I-upload ang PDF file na nais mong idagdag\n3. Ipasok ang pamagat ng proseso\n4. Pindutin ang Upload! button\n\nPaalala: Siguraduhing hindi isinasabay ang pag-upload ng link at PDF file nang sabay-sabay.",
        "how": "Paano Gamitin"
    },
    "Cebuano": {
        "problem": "Ang Problema",
        "prob_desc": "Ang pagsabot ug pag-navigate sa mga proseso sa gobyerno kasagaran lisud ug makahurot sa panahon, nga mosangpot sa kalibog ug kasagmuyo. Ang kakulang sa dali nga ma-access, tin-aw, ug mubo nga impormasyon nakatampo sa pagka-inefficiency ug ubos nga civic engagement, ilabi na alang niadtong adunay limitado nga mga kapanguhaan ug niadtong nag-atubang sa mga babag sama sa kahanas sa pinulongan ug visual impairment.",
        "sol": "Ang Solusyon",
        "sol_desc": "Ang among aplikasyon nagpadali sa mga proseso sa gobyerno pinaagi sa paghatag ug usa ka user-friendly nga plataporma diin ang usa dali ug dali nga makapangutana ug makakuha og may kalabutan nga impormasyon sa lain-laing mga pinulongan.",
        "feat": "Mga Bahin",
        "feat_titles": ["Interactive nga Chatbot", "Suporta sa Daghang Pinulongan", "Text-to-Speech"],
        "feat_desc": ["Kontaka ang usa ka chatbot aron makakuha dayon nga mga tubag ug giya sa lainlaing mga proseso.", "Pagkuha og impormasyon sa imong gipili nga pinulongan.", "Paminaw sa impormasyon nga gibasa og kusog para sa mas sayon ​​nga pagsabot."],
        "mission": "Atong Misyon",
        "mis_desc": "Among gihatagan og gahom ang mga lungsoranon pinaagi sa pagpahapsay sa mga proseso sa gobyerno, pagpasiugda sa transparency, ug pagpakusog sa partisipasyon sa sibiko. Ang among tumong mao ang paghimo sa impormasyon gikan sa importanteng mga serbisyo sa gobyerno nga ma-access ug masabtan sa tanan. Pinaagi sa kabag-ohan ug kooperasyon sa komunidad, gipadali namo ang maayong paghimog desisyon ug aktibong partisipasyon sa sibiko.",
        "start_title": "Giya sa Tiggamit",
        "query_title": "Panid sa Pangutana",
        "query_desc": "Dali nga pagpangita og impormasyon bahin sa lain-laing mga proseso sa gobyerno, sama sa pag-aplay alang sa lisensya sa pagmaneho ug pagrehistro sa mga botante. Kini nga panid nagsuporta sa lain-laing mga pinulongan ug ang mga resulta mahimong basahon sa kusog (Text to Speech) aron mahimo kini nga mas sayon.",
        "query_how_to": f"1. Pili-a ang proseso nga imong gusto'ng masayran og dugang kabahin niini.\n2. I-type ang imong pangutana sa bar nga 'Magpangutana' nga anaa sa ubos sa screen.\n3. Pinduta ang send button sa pinakadangpuon nga bahin o pinduta ang 'Enter' key sa keyboard.\n4. Hulati ang tubag sa aplikasyon. Mahinungdanon nga mohulat kini sa pipila ka mga segundo.\n5. Magpangutana og bisan unsa ka daghang pangutana nga imong gusto.",
        "data_title": "Idugang sa Database",
        "data_desc": "Ang mga manedyer mahimong makadugang o maka-update sa impormasyon bahin sa lainlaing mga proseso. Kini siguradong nagtipig niini nga bag-o, komprehensibo, ug nagpakita sa pinakabag-o nga mga teknik, nga nagpadayon sa aplikasyon nga may kalabutan ug mapuslanon.",
        "data_how_to": "Kon gusto ka muadto sa usa ka webpage ngadto sa database:\n\n1. Butangi ang link sa text area\n2. Ipasulod ang pamagat sa proseso\n3. Pinduta ang Upload! button\n4. Tan-awa ang database aron makita kon kini'y na-successfully nga gidugang.\n\nKon gusto ka muadto sa usa ka PDF file:\n1. I-click ang 'Upload PDF'\n2. I-upload ang PDF file nga imong gusto idugang\n3. Ipasulod ang pamagat sa proseso\n4. Pinduta ang Upload! button\n\nPaalala: Siguraduha nga dili imong isabay ang pag-upload sa link ug PDF file nga sabay-sabay.",
        "how": "Unsaon Paggamit"
    },
    "Hiligaynon": {
        "problem": "Ang Problema",
        "prob_desc": "Madamo nga kabudlayan kag pag-antos ang nagaabot kon matun-an kag masudlan ang mga proseso sa gobyerno. Ang kakulangan sang madali ma-access, klaro, kag maikli nga impormasyon nagadugang sa kawalay epektibo kag gindugangan ang limitasyon sang pag-ambit sa komunidad, ilabi na gid sa mga wala sing sapat nga mga mapag-on kag ang mga naatubang sang mga balaghal sa paglantaw kag mga kakulangan sa pagtulok.",
        "sol": "Ang Solusyon",
        "sol_desc": "Ang amon nga aplikasyon nagapahapos sang mga proseso sa gobyerno paagi sa paghatag sang user-friendly nga plataporma diin ang mga ginagamit madasig nga makapangayo kag makuha ang mga importante nga impormasyon sa madamo nga lenguahe.",
        "feat": "Mga Katuyuan",
        "feat_titles": ["Interaktibo nga Chatbot", "Suporta sa Madamo nga Lenguahe", "Text-to-Speech"],
        "feat_desc": ["Makig-angot sa chatbot para makakuha sang pasadlawan nga mga tublag kag panuytoy sa iban nga mga proseso.", "Makakuha sang impormasyon sa imo nga ginapili nga lenguahe.", "Pamati sa impormasyon nga ginabasa para mas madali ang pag-intiendi."],
        "mission": "Amon Misyon",
        "mis_desc": "Ginapalakat namon ang mga pumuluyo sa pagpapadali sang mga proseso sang gobierno, pagpanguna sa pagkalinaw, kag pagpapalig-on sang pag-ambit sa sibika. Ang amon tuyo amo ang paghimulag sang pagkamadaug kag pagsabat sang tanan nga impormasyon parte sa mga esensyal nga serbisyo sang gobierno para sa tanan. Paagi sa pag-inobat kag pagsaligay sa komunidad, aton man ginasuportahan ang mas maayo nga pagdesisyon kag aktibo nga pag-ambit sa sibika.",
        "start_title": "Giya",
        "query_title": "Mga Pamangkot",
        "query_desc": "Madali nga magpangita sang impormasyon parte sa iban-iban nga mga proseso sang gobierno, pareho sang aplikasyon sang lisensya sang drayber kag pagrejistro sang mga botante. Ang pahina nagasuporta sa madamo nga lenguahe kag may kasulhayan nga text-to-speech para sa mas maayo nga pag-abot.",
        "query_how_to": "1. Pili-a ang proseso nga gusto mo masabat pa nga detalyado.\n2. I-type ang imo nga pangutana sa 'Magpangutana' nga bar sa idalom sang screen.\n3. Pinduta ang send button sa pinakadangpan nga bahin ukon pinduton ang 'Enter' key sa keyboard.\n4. Hulata ang tublag sang aplikasyon. Mahimo nga magdugay ini sa pila ka segundo.\n5. Pangutana sang bisan ano nga damo mo nga gusto.",
        "data_title": "Idugang sa Database",
        "data_desc": "Ang mga administrador puede mag-amot ukon mag-update sang impormasyon parte sa mga proseso sang gobierno. Ini nagasiguro nga ang database mas aktwal, bug-os, kag nagareplekta sang pinakabag-o nga mga pamaagi, nga nagapadayon sang pagkainabyanay kag kapuslanan sang aplikasyon.",
        "data_how_to": "Kon gusto mo magdugang sang isa ka webpage sa database:\n\n1. Ibutang ang link sa text area\n2. Ipasulod ang titulo sang proseso\n3. Pinduton ang Upload! button\n4. Tan-awa ang database agod makita kon matagumpay nga nadugangan.\n\nKon gusto mo magdugang sang isa ka PDF file:\n1. I-click ang 'Upload PDF'\n2. I-upload ang PDF file nga gusto mo idugang\n3. Ipasulod ang titulo sang proseso\n4. Pinduton ang Upload! button\n\nPaalala: Siguraduha nga wala ginahalinan ang pag-upload sang link kag PDF file nga sabay-sabay.",
        "how": "Paano Gamiton"
    },
    "Ilocano": {
        "problem": "Ti Parikut",
        "prob_desc": "Masansan a narigat ken makabusbos iti panawen ti pannakaawat ken panangiturong kadagiti proseso ti gobierno, a pakaigapuan ti pannakariro ken panagdanag. Ti kaawan ti nalaka a magun-od, nalawag, ken ababa nga impormasion ket makatulong iti kinaawan epektibo ken nababbaba a pannakipaset ti sibiko, nangruna kadagidiay addaan kadagiti limitado a rekurso ken dagidiay addaan kadagiti lapped a kas ti abilidad ti pagsasao ken dagiti parikut iti panagkita.",
        "sol": "Ti Solusion",
        "sol_desc": "Ti aplikasionmi ket mangpasayaat kadagiti proseso ti gobierno babaen ti panangipaay iti nalaka nga usaren a plataporma a sadiay ti maysa ket nalaka ken napardas nga agsaludsod ken makagun-od kadagiti mainaig nga impormasion iti nadumaduma a pagsasao.",
        "feat": "Features",
        "feat_titles": ["Interactive Chatbot", "Suporta iti Multilingual", "Text-to-Speech"],
        "feat_desc": ["Makilangen iti chatbot tapno makagun-od iti dagus a sungbat ken panangiwanwan kadagiti nadumaduma a proseso.", "Mangala iti impormasion iti pagsasao a piliem.", "Dumngeg iti impormasion a mabasa tapno nalaklaka a maawatan."],
        "mission": "Misiontayo",
        "mis_desc": "Pabilgenmi dagiti umili babaen ti panangpasayaat kadagiti proseso ti gobierno, panangitandudo iti kinalawag, ken panangpakaro iti pannakipaset dagiti sibiko. Ti panggepmi ket pagbalinen a magun-od ken maawatan ti amin ti impormasion manipud kadagiti napateg a serbisio ti gobierno. Babaen ti panagbalbaliw ken panagtitinnulong ti komunidad, mapasayaatmi ti nasayaat a panagaramid ti desision ken aktibo a pannakipaset ti sibiko.",
        "start_title": "Giya ti Agus-usar",
        "query_title": "Panid ti Saludsod",
        "query_desc": "Nalaka laeng a masarakan ti impormasion maipapan iti nadumaduma a proseso ti gobierno, kas iti panagaplay iti lisensia ti panagmaneho ken panagrehistro iti botante. Daytoy a panid ket mangsuporta kadagiti nadumaduma a pagsasao ken dagiti resulta ket mabalin a mabasa iti napigsa (Text to Speech) tapno ad-adda a makastrek.",
        "query_how_to": "1. Pilien ti proseso a kayatmo a maammuan ti ad-adu pay maipapan iti dayta.\n2. I-type ti saludsodmo iti ‘Ask a question’ bar iti baba ti iskrin.\n3. Pinduten ti send button iti akinkannawan wenno i-press ti 'Enter' key iti keyboard.\n4. Urayen ti sungbat ti aplikasion. Mabalin nga apagbiit laeng daytoy.\n5. Agsaludsodka agingga a kayatmo.",
        "data_title": "Inayon iti Database",
        "data_desc": "Mabalin nga inayon wenno i-update dagiti manager ti impormasion maipapan kadagiti nadumaduma a proseso. Sigurado a pagtalinaedenna dayta nga agdama, komprehensibo, ken iyanninawna dagiti kabaruan a teknik, a mangtaginayon iti aplikasion a mainaig ken makagunggona.",
        "data_how_to": "No kayatmo ti manginayon ti panid ti web iti database:\n\n1. Isurat ti link iti text area\n2. Isurat ti paulo ti proseso\n3. Pinduten ti Upload! buton\n4. Kitaen ti database tapno makita no sibaballigi a nainayon dayta\n\nNo kayatmo ti mangnayon iti PDF file:\n1. I-click ti 'I-upload ti PDF'.\n2. I-upload ang PDF file na nais mong idagdag\n3. Isurat ti paulo ti proseso\n4. Pinduten ti Upload! buton\n\nPakaammo: Siguraduen a saan nga aggigiddan nga i-upload ti link ken PDF file.",
        "how": "Kasano ti Agusar"
    }
}

#####     FUNCTIONS     #####
def get_process_text(url):
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    }
    try:
        r = session.get(url, headers=headers)
    except requests.exceptions.ConnectionError as e:
        # Site has untrusted SSL
        print("Warning... Untrusted SSL")
        r = session.get(url, headers=headers, verify=False)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup.text.replace('error: Content is protected !!', '').replace('\t', ' ').replace('\n\n', '').strip()


def get_openai_client():
    """Function to create an OpenAI client"""
    client = OpenAI(api_key=OPENAI_APIKEY)
    return client

def get_splitted_documents_df(df, chunk_size=2000, chunk_overlap=300):
    """Function to split the text into documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    documents = text_splitter.create_documents(
        texts = df['text'],
        metadatas = [{"url": url, "process_title": title, "version": ver} for url, title, ver in df[['url', 'process_title', 'version']].values]
    )

    return documents

def init_chroma_db(collection_name, db_path='gov_process.db'):
    """Function to intialize the database"""
    # Create a Chroma Client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Create an embedding function
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_APIKEY, model_name=EMBEDDING_MODEL)

    # Create a collection
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    return collection

def upsert_documents_to_collection(documents, collection):
    """Function to update or insert documents into the collection"""
    # Every document needs an id for Chroma
    last_idx = len(collection.get()['ids'])
    ids = list(f'id_{idx+last_idx:010d}' for idx, _ in enumerate(documents))
    docs = list(map(lambda x: x.page_content, documents))
    mets = list(map(lambda x: x.metadata, documents))

    # Update/Insert some text documents to the db collection
    collection.upsert(ids=ids, documents=docs,  metadatas=mets)

def add_gov_process_to_chroma(collection, url, process_title, ver, isPDF = False):
    """Function to update the database"""
    if isPDF:
        df = pd.DataFrame({'text': [url], 'process_title': [process_title], 'url': "PDF File", 'version': [ver]})
    else:
        # Step 1: Scrape the text content
        text = get_process_text(url)

        # Step 2: Turn into dataframe
        df = pd.DataFrame({'text': [text], 'process_title': [process_title], 'url': [url], 'version': [ver]})

    # Step 3: Split the text into chunks
    docs = get_splitted_documents_df(df, chunk_size=2000, chunk_overlap=300)
    
    # Step 4: Upsert the documents into the Chroma database collection
    upsert_documents_to_collection(docs, collection)

    return df, docs

def process_semantic_search(Q, metadata_key, meta_val, k=3, collection=None):
    """Function to query a subset of the collection (based on a metadata)"""
    results = collection.query(
        query_texts=[Q],    # Chroma will embed this for you
        n_results=k,        # How many results to return,
        where={f"{metadata_key}": f"{meta_val}"} # specific data only
    )
    return results

def latest_semantic_search(Q, title, ver=0, k=3, collection=None):
    """Function to query a subset of the collection (based on the title and version)"""
    results = collection.query(
        query_texts=[Q],    # Chroma will embed this for you
        n_results=k,        # How many results to return,
        where = {
            "$and": [
                {"process_title": f"{title}"},
                {"version": ver}
            ]
        }
    )
    
    return results

def generate_response(task, prompt, llm):
    """Function to generate a response from the LLM given a specific task and user prompt"""
    response = llm.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': f"Perform the specified task: {task}"},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def generate_translation(doc, llm, from_lang = "English", to_lang="Tagalog"):
    """Function to translate a document from English to Language of Choice"""
    task = 'Text Translation'
    prompt = f"""Translate this document from {from_lang} to {to_lang}:\n\n{doc}
    Just give the direct translation and don't add anything else to the response.
    """
    response = generate_response(task, prompt, llm)

    return response

def generate_step_by_step(Q, text, llm):
    """Function to generate the step-by-step process"""
    prompt = f"""
    Provide the step-by-step process on {Q} based on this guidelines:\n\n{text}. Your response should only contain the step-by-step process.
    """
    response = generate_response(Q, prompt, llm)
    return response

def generate_response_to_question(Q, text, title, llm):
    """Generalized function to answer a question"""
    prompt = f"""
    Provide the answer on {Q} based on {title} given this guidelines:\n\n{text}.

    You should only respond based on the given process and guideline. Don't respond if you don't know the answer. Don't give more than what is asked. Only answer the questions directly related to the {title} and the given guidelines. If not directly stated in the guidelines, say that and don't give assumptions.
    """
    response = generate_response(Q, prompt, llm)
    return response

def ask_query(Q, title, ver, llm, k=7):
    """Function to go from question to query to proper answer"""
    # Get related documents
    query_result = latest_semantic_search(Q=Q, title=title, ver=ver, k=k, collection=collection)

    # Get the text of the documents
    text = query_result['documents'][0][0]

    # Pass into GPT to get a better formatted response to the question
    response = generate_response_to_question(Q, text, title, llm=llm)
    # return Markdown(response)
    return response

def get_dataframe(collection, latest=False):
    """Function to get all the titles and urls from the collection and returns a dataframe"""
    metadatas = collection.get()['metadatas']
    database = []

    for meta in metadatas:
        if meta not in database:
            database.append(meta)

    df = pd.DataFrame(database)

    # If only want to get the latest versions of the processes in the collection
    if latest:
        df.sort_values(by='version', inplace=True)
        df.drop_duplicates(subset="process_title", keep='last', inplace=True)
    
    # return database
    return df.reset_index().drop(columns=['index'], axis=1)

def new_process(url, title, collection, isPDF = False):
    """Function to add a new process into the database"""
    df = get_dataframe(collection)
    ver = 0
    
    try:
    # If using the same title as an existing one, update the version
        if title in list(df['process_title']):
            ver = df[df['process_title'] == title]['version'].max() + 1
        # If the link exists in the database, just get a new version
        if url in list(df['url']) and title not in list(df['process_title']):
            ver = df[df["url"]==url]['version'][0] + 1
    except KeyError:
        pass

    # Update the database
    if isPDF:
        add_gov_process_to_chroma(collection, url, title, ver, True)
    else:
        add_gov_process_to_chroma(collection, url, title, ver)

def text_to_speech(text, lang='en'):
    """Function to make the app "speak" based on the text"""
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(mp3_fp)
    sound = mp3_fp
    sound.seek(0)
    
    # Convert speech to base64 encoding
    b64 = base64.b64encode(sound.read()).decode('utf-8')

    md = f"""
        <audio id="audioTag" controls autoplay>
        <source src="data:audio/mp3;base64,{b64}"  type="audio/mpeg" format="audio/mpeg">
        </audio>
        """
    
    st.markdown(
        md,
        unsafe_allow_html=True,
    )

#####     BTS Contants     #####
# Initialize database
collection = init_chroma_db(collection_name="gov_process", db_path="gov_process.db")

# Initialize OpenAI Client
llm = get_openai_client()

##### STYLED COMPONENTS #####
# Set universal styles
html_styles = f"""
<style>
.smaller-text {{
    font-size: 14px;
    margin-bottom: 10px;
}}
</style>
<p>Insert a tagline</p>
"""

def suggestions_bar(questions):
    suggest_html = f"""
    <style>
        .suggest-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin: auto;
            align-items: stretch;
        }}
        
        .question {{
            flex: 32%;
            padding: 10px 20px;
            border: 1px, solid, {COLOR_GRAY};
            border-radius: 10px;
            background-color: {COLOR_GRAY};
            justify-content: center;
            align-items: center;
            display: flex;
            flex-direction: column;
        }}

        .question p {{
            text-align: center;
            margin: 0;
        }}

        .smaller-text {{
            font-size: 14px;
            margin-bottom: 10px;
        }}
    </style>
    <p class='smaller-text' id="suggested-qs">Suggested Questions</p>
    <div class="suggest-container">
        <div class='question'>
            <p>{questions[0]}</p>
        </div>
        <div class='question'>
            <p>{questions[1]}</p>
        </div>
        <div class='question'>
            <p>{questions[2]}</p>
        </div>
    </div>
    """
    st.markdown(suggest_html, unsafe_allow_html=True)

scroll_back_to_top_btn = f"""
<style>
    .scroll-btn {{
        position: absolute;
        border: 2px solid #31333f;
        background: {COLOR_GRAY};
        border-radius: 10px;
        padding: 2px 10px;
        bottom: 0;
        right: 0;
    }}

    .scroll-btn:hover {{
        color: #ff4b4b;
        border-color: #ff4b4b;
    }}
</style>
<a href="#government-processease">
    <button class='scroll-btn'>
        Back to Top
    </button>
</a>
"""

# Homepage
def home():
    # Get the text variables in the proper language
    problem = homepage_titles[lang]["problem"]
    prob_desc = homepage_titles[lang]["prob_desc"]
    sol = homepage_titles[lang]["sol"]
    sol_desc = homepage_titles[lang]["sol_desc"]
    feat = homepage_titles[lang]["feat"]
    feat_titles = homepage_titles[lang]["feat_titles"]
    feat_desc = homepage_titles[lang]["feat_desc"]
    mission = homepage_titles[lang]["mission"]
    mis_desc = homepage_titles[lang]["mis_desc"]
    start_title = homepage_titles[lang]["start_title"]
    query_title = homepage_titles[lang]["query_title"]
    query_desc = homepage_titles[lang]["query_desc"]
    query_how_to = homepage_titles[lang]["query_how_to"]
    data_title = homepage_titles[lang]["data_title"]
    data_desc = homepage_titles[lang]["data_desc"]
    data_how_to = homepage_titles[lang]["data_how_to"]
    how = homepage_titles[lang]["how"]

    homepage_html = f"""
    <style>
        .title-container {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        .subheader {{
            text-align: center;
            margin-top: 3rem;
            margin-bottom: 0.5rem;
            font-size: calc(1.3rem + .6vw);
            font-weight: 700;
        }}

        .two-col-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 3rem;
            justify-content: center;
            align-items: center;
            margin: 3rem 1rem;
        }}

        .two-col-container p, .two-col-container h3 {{
            margin: 0;
            padding: 0;
            text-align: justify;
        }}

        .left {{
            flex: 40%;
        }}

        .right {{
            flex: 50%;
        }}

        .features {{
            border: 2px solid {COLOR_RED};
            border-radius: 20px;
            padding-bottom: 4rem;
            padding: auto 2rem;
        }}

        .feat-cards {{
            display: flex;
            gap: 2rem;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            padding: 0 1rem;
        }}

        .feat-card {{
            text-align: center;
            flex: 30%;
            margin: auto;
        }}

        .feat-card img {{
            border: 1px solid black;
            border-radius: 50%;
            padding: 10px;
        }}

        .feat-card p {{
            text-align: center;
        }}

        .feat-title {{
            font-size: 22px;
            font-weight: 600;
            margin-top: 1rem;
        }}

        .icon-creds {{
            font-size: 14px;
            display: inline;
            
        }}

        .mis-desc {{
            margin: auto 2.5rem;
            text-align: center;

        }}
    </style>
    <div class='title-container'>
        <h1 class="home-title">{APP_NAME}</h1>
        <p class='tagline'>{APP_DESC}</p>
    </div>
    <div class='two-col-container'>
        <h3 class='left subheader'>{problem}</h3>
        <p class='right desc'>{prob_desc}</p>
    </div>
    <div class='two-col-container'>
        <h3 class='left subheader'>{sol}</h3>
        <p class='right desc'>{sol_desc}</p>
    </div>
    <div class='features'>
        <p class='subheader'>{feat}</p>
        <div class='feat-cards'>
            <div class='feat-card'>
                <img width="75" height="75" src="https://img.icons8.com/external-vectorslab-detailed-outline-vectorslab/68/external-Coding-Chat-mobile-app-development-vectorslab-detailed-outline-vectorslab.png" alt="external-Coding-Chat-mobile-app-development-vectorslab-detailed-outline-vectorslab"/>
                <p class='feat-title'>{feat_titles[0]}</p>
                <p class='feat-desc'>{feat_desc[0]}</p>
                <a class='icon-creds' href="https://icons8.com/icon/BtQRmgfU7t43/chatbot-development">Chatbot Development</a><p class='icon-creds'> icon by </p><a class='icon-creds' href="https://icons8.com">Icons8</a>
            </div>
            <div class='feat-card'>
                <img width="75" height="75" src="https://img.icons8.com/ios/50/translation.png" alt="translation"/>
                <p class='feat-title'>{feat_titles[1]}</p>
                <p class='feat-desc'>{feat_desc[1]}</p>
                <a class='icon-creds' href="https://icons8.com/icon/10728/translation">Translation</a><p class='icon-creds'> icon by </p><a class='icon-creds' href="https://icons8.com">Icons8</a>
            </div>
            <div class='feat-card'>
                <img width="75" height="75" src="https://img.icons8.com/fluency-systems-regular/48/speech-to-text.png" alt="speech-to-text"/>
                <p class='feat-title'>{feat_titles[2]}</p>
                <p class='feat-desc'>{feat_desc[2]}</p>
                <a class='icon-creds' href="https://icons8.com/icon/JyVJOmkgQcg4/speech-to-text">Speech To Text</a><p class='icon-creds'> icon by </p><a class='icon-creds' href="https://icons8.com">Icons8</a>
            </div>
        </div>
    </div>
    <p class='subheader'>{mission}</p>
    <p class='mis-desc'>{mis_desc}</p>
    <p class='subheader'>{start_title}</p>
    """

    query_style = f"""
    <style>
        .sub-subheader {{
            text-align: center;
            margin-bottom: 0.5rem;
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .sub-subheader-text {{
            font-size: 1rem;
            text-align: center;
            margin: auto 2rem;
        }}
    </style>
    <p class='sub-subheader'>{query_title}</p>
    <p class='sub-subheader-text'>{query_desc}</p>
    <br>
    """

    data_style = f"""
    <p class='sub-subheader'>{data_title}</p>
    <p class='sub-subheader-text'>{data_desc}</p>
    <br>
    """

    st.markdown(homepage_html, unsafe_allow_html=True)
    query_info, data_info = st.columns(2, gap='medium', vertical_alignment='center')
    with query_info:
        st.markdown(query_style, unsafe_allow_html=True)
        with st.expander(f"ℹ️ {how}"):
            st.markdown(query_how_to)
    with data_info:
        st.markdown(data_style, unsafe_allow_html=True)
        with st.expander(f"ℹ️ {how}"):
            st.markdown(data_how_to)

# Query Page
def query_page():
    st.title(APP_NAME)
    st.write("___")
    df = get_dataframe(collection, latest=True)
    orig_titles = list(df['process_title'])

    if lang != "English":
        titles = list(map(lambda x: generate_translation(x, llm, "English", lang), orig_titles))
        title_label = generate_translation("Choose a government process", llm, "English", lang)
        user_prompt = generate_translation("Ask a question", llm, "English", lang)
        load_response = generate_translation("Loading a response...", llm, "English", lang)
    else:
        titles = orig_titles
        title_label = 'Choose a government process'
        user_prompt = "Ask a question"
        load_response = "Loading a response..."
    
    title = st.selectbox(label=title_label, options=titles)
    index = titles.index(title) # get the index incase the title is not in eng
    process_to_focus = orig_titles[index] # get the actual title (in eng)
    ver = list(df[df['process_title'] == process_to_focus]['version'])[0]

    # Show suggested questions
    if process_to_focus in suggested_questions.keys():
        question_list = suggested_questions[process_to_focus]
        random_suggested_questions = set()
        while len(random_suggested_questions) < 3:
            random_suggested_questions.add(random.choice(question_list))
        
        suggestions_bar(list(random_suggested_questions))
    
    st.write("___")
    
    # Set a default model
    if 'openai_model' not in st.session_state:
        st.session_state['openai_model'] = 'gpt-4o'

    # Initialize title
    if "title" not in st.session_state:
        st.session_state['title'] = ""

    # Initialize chat history or reset history if you change
    if "messages" not in st.session_state or st.session_state['title'] != title:
        st.session_state.messages = []
        st.session_state['title'] = title
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
                
    # Accept user input
    if prompt := st.chat_input(user_prompt):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user",
                                            "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Translate the user prompt to english
        if lang != "English":
            prompt = generate_translation(prompt, llm, lang, "English")
        
        # Display response
        with st.chat_message("assistant"):
            with st.spinner(load_response):
                response = ask_query(prompt, process_to_focus, ver, llm, k=7)

                # Translate the response if the language of choice is not in English
                if lang != "English":
                    response = generate_translation(response, llm, "English", lang)

            st.markdown(response)

            st.session_state.messages.append({"role": "assistant",
                                                "content": response})

            # Text to Speech the response (if enabled)
            if speak:
                if lang == 'English':
                    text_to_speech(response, lang='en')
                else:
                    # tl = Filipino (the only one available)
                    text_to_speech(response, lang='tl')
            
        
        # Add a scroll back to top button
        st.markdown(scroll_back_to_top_btn, unsafe_allow_html=True)

# Upload Page
def upload_page():
    st.title(APP_NAME)
    st.write("___")
    st.subheader("Add to Database")

    df = pd.DataFrame()
    upload_type = ""
    text = ""
    url_tab, pdf_tab = st.tabs(["Upload Link", "Upload PDF"])

    # Get the link/file and title from the user
    with url_tab:
        url = st.text_area("Enter URL:", placeholder="Enter the URL of the process you want to upload").strip()
        upload_type = 'url'
    with pdf_tab:
        uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
        if uploaded_file is not None:
            upload_type = 'pdf'
            # Create PDF reader
            reader = PdfReader(uploaded_file)
            text = ""

            # Extract Text from each page
            for page in reader.pages:
                text += page.extract_text()

    title = st.text_area("Enter process title:", placeholder="Enter the title of the process you are adding").strip()

    # Update collection
    if st.button("Upload!") and (url != "" or text != "") and title != "":
        if upload_type == 'url':
            new_process(url, title, collection, False)
            # st.toast
        else:
            url = " ".join(text.split())
            new_process(url, title, collection, True)
    
        st.toast("Successfully added to database!", icon='🎉')

    # Showing database to the users (processes available)
    whole_tab, unique_tab = st.tabs(["View Whole Database", "View Unique Processes"])

    with whole_tab:
        df = get_dataframe(collection)
        st.dataframe(df)

    with unique_tab:
        df = get_dataframe(collection, latest=True)
        st.dataframe(df)
    

#####     MAIN SITE     #####
# Create streamlit app
st.set_page_config(layout='wide')

homepage = st.Page(home, title="Home", icon=":material/home:")
querypage = st.Page(query_page, title="Query Processes", icon=":material/search:")
uploadpage = st.Page(upload_page, title="Add to Database", icon=":material/upload_file:")

pg = st.navigation(
    {
        "Navigation": [homepage, querypage, uploadpage]
    }
)

with st.sidebar.expander("⚙️ Response Settings"):
    lang = st.selectbox(
        "Language Options", ["English", "Tagalog", "Cebuano", "Hiligaynon", "Ilocano"]
    )
    speak = st.toggle("Text to Speech")

pg.run()