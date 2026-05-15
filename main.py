import streamlit as st
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')

# Inne przykłady do wypróbowania:
# st.balloons() # animowane balony ;)
# st.error('Błąd!') # wyświetla informację o błędzie
# st.warning('Ostrzeżenie, działa, ale chyba tak sobie...')
# st.info('Informacja...')
# st.success('Udało się!')

st.spinner()
with st.spinner(text='Pracuję...'):
    time.sleep(1)

st.title('Lab05. Streamlit')
# title, jak sama nazwa wskazuje, używamy do wyświetlenia tytułu naszej aplikacji

st.header('Wprowadzenie do zajęć')
# header to jeden z podtytułów wykorzystywnaych w Streamlit

st.subheader('O Streamlit')
# subheader to jeden z podtytułów wykorzystywnaych w Streamlit

st.text('To przykładowa aplikacja z wykorzystaniem Streamlit')
# text używamy do wyświetlenia dowolnego tekstu. Można korzystać z polskich znaków.

st.write('Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego.')
# write używamy również do wyświetlenia tekstu, różnica polega na formatowaniu.

st.code("st.write()", language='python')
# code może nam się czasami przydać, jeżeli chcielibyśmy pokazać np. klientowi fragment kodu, który wykorzystujemy w aplikacji

with st.echo():
    st.write("Echo")
# możemy też to zrobić prościej używając echo - pokazujemy kod i równocześnie go wykonujemy

df = pd.read_csv("DSP_4.csv", sep = ';')
st.dataframe(df)
# musimy tylko pamiętać o właściwym określeniu separatora (w tym wypadku to średnik)
# masz problem z otworzeniem pliku? sprawdź w jakim katalogu pracujesz i dodaj tam plik (albo co bardziej korzystne - zmień katalog pracy)
# os.getcwd() # pokaż bieżący katalog
# os.chdir("") # zmiana katalogu

st.header('Przetwarzanie języka naturalnego')

import streamlit as st
from transformers import pipeline

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie tekstu z angielskiego na niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)

elif option == "Tłumaczenie tekstu z angielskiego na niemiecki":
    with st.expander("ℹ️ Instrukcja obsługi aplikacji (Kliknij, aby rozwinąć)"):
        st.write("""
        W formularzu poniżej wpisz tekst po angielsku który chcesz przetłumaczyć na język niemiecki i kliknij Ctrl+Enter.
        """)
    st.image("de_translation_banner.jpg", use_container_width=True)
    text = st.text_area(label="Wpisz tekst do przetłumaczenia (w j. angielskim):", placeholder="Wprowadź tekst...")
    if text:
        with st.spinner("Trwa tłumaczenie na język niemiecki..."):
            tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
            model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
            inputs = tokenizer(text, return_tensors="pt", padding=True)

            translated = model.generate(**inputs)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
        st.success("Tłumaczenie gotowe!")
        st.write("**Przetłumaczony tekst:**")
        st.info(result)

st.write('---')
st.write("Nr. indeksu s24821")
st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')
