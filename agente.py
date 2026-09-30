"""
Agente de pesquisa acadêmica — Minicurso "Introdução aos Agentes de IA" (XV SINFO)

Pedido de exemplo:
    "Resuma os 3 artigos mais recentes sobre RAG e envie para minha orientadora."

Componentes (os mesmos dos slides):
    LLM .............. modelo aberto via API do Hugging Face (Inference Providers)
    Instruções ....... prompt de sistema com papel, objetivo, regras e limites
    Ferramentas ...... buscar_artigos (API do arXiv), buscar_contato (base de contatos),
                       enviar_whatsapp (WhatsApp Web, sem API), enviar_email (simulado)
    Memória .......... curto prazo: histórico da conversa (checkpointer + thread_id)
                       longo prazo: contatos.json
    Limite de autonomia: nada é enviado sem um humano aprovar
                       (WhatsApp: você clica em Enviar; e-mail: aprovação no terminal)

Execução:
    pip install -r requirements.txt
    cp .env.example .env        # coloque seu token do Hugging Face
    python agente.py
"""

import json
import os
import smtplib
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()
PASTA = Path(__file__).parent


# ---------------------------------------------------------------------------
# ETAPA 1 — Integração com o modelo de linguagem (LLM)
# ---------------------------------------------------------------------------
# Qualquer modelo de chat com suporte a "tool calling" nos Inference Providers
# do Hugging Face funciona. Troque pelo que estiver disponível na sua conta.
MODELO = os.getenv("HF_MODELO", "Qwen/Qwen2.5-72B-Instruct")


def criar_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=MODELO,
        task="text-generation",
        provider="auto",          # o Hugging Face escolhe um provedor disponível
        max_new_tokens=1024,
        temperature=0.1,          # respostas mais estáveis
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )
    return ChatHuggingFace(llm=endpoint)


# ---------------------------------------------------------------------------
# ETAPA 2 — Instruções e objetivo (prompt de sistema)
# ---------------------------------------------------------------------------
INSTRUCOES = """Você é um assistente de pesquisa acadêmica.

Objetivo: encontrar artigos científicos, resumi-los e, quando o usuário pedir,
enviar o resumo para alguém.

Regras:
1. Use a ferramenta buscar_artigos para encontrar artigos. Nunca invente títulos,
   autores, datas ou links.
2. Pesquise em inglês (ex.: "retrieval augmented generation" para RAG).
3. Resuma cada artigo em 2 ou 3 frases, em português, com título, data e link.
4. Para descobrir o contato de alguém, use buscar_contato. Se não encontrar,
   peça o telefone ou o e-mail ao usuário.
5. Envie pelo canal que o usuário pedir. Se ele não disser, use o
   canal_preferido do contato ("whatsapp" ou "email").
6. No WhatsApp, envie uma versão curta: título e link de cada artigo.
7. Só envie depois que o resumo estiver pronto.
8. Responda sempre em português."""


# ---------------------------------------------------------------------------
# ETAPA 3 — Ferramenta: busca de artigos (API externa do arXiv)
# ---------------------------------------------------------------------------
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _texto(elemento, caminho):
    return " ".join((elemento.findtext(caminho, "", ATOM) or "").split())


@tool
def buscar_artigos(tema: str, quantidade: int = 3) -> str:
    """Busca no arXiv os artigos mais recentes sobre um tema (em inglês).
    Retorna título, autores, data, link e resumo (abstract) de cada artigo."""
    parametros = {
        "search_query": f'all:"{tema}"',
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max(1, min(int(quantidade), 10)),
    }
    resposta = requests.get(ARXIV_API, params=parametros, timeout=30)
    resposta.raise_for_status()
    raiz = ET.fromstring(resposta.text)

    artigos = []
    for entrada in raiz.findall("a:entry", ATOM):
        artigos.append({
            "titulo": _texto(entrada, "a:title"),
            "autores": [_texto(a, "a:name") for a in entrada.findall("a:author", ATOM)][:3],
            "data": _texto(entrada, "a:published")[:10],
            "link": _texto(entrada, "a:id"),
            "resumo": _texto(entrada, "a:summary")[:1500],
        })
    if not artigos:
        return "Nenhum artigo encontrado para esse tema."
    return json.dumps(artigos, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ETAPA 4 — Integração com uma base de dados: contatos (memória de longo prazo)
# ---------------------------------------------------------------------------
ARQUIVO_CONTATOS = PASTA / "contatos.json"


def carregar_contatos() -> dict:
    if ARQUIVO_CONTATOS.exists():
        return json.loads(ARQUIVO_CONTATOS.read_text(encoding="utf-8"))
    return {}


@tool
def buscar_contato(nome: str) -> str:
    """Procura o e-mail de uma pessoa na agenda de contatos do usuário.
    Aceita o nome ou o papel da pessoa (ex.: 'orientadora')."""
    busca = nome.strip().lower()
    for chave, contato in carregar_contatos().items():
        if busca in chave.lower() or busca in contato.get("nome", "").lower():
            return json.dumps(contato, ensure_ascii=False)
    return f"Nenhum contato encontrado para '{nome}'."


# ---------------------------------------------------------------------------
# ETAPA 5 — Ferramentas de ação: WhatsApp Web e e-mail
# ---------------------------------------------------------------------------
# WhatsApp sem API: abre o WhatsApp Web com a mensagem pronta.
# O envio final é o clique do usuário em "Enviar" (humano no circuito).
@tool
def enviar_whatsapp(telefone: str, mensagem: str) -> str:
    """Abre o WhatsApp Web com a mensagem pronta para o contato.
    telefone: com código do país e DDD, só números (ex.: 5586999999999).
    mensagem: curta, com título e link de cada artigo."""
    numero = "".join(c for c in telefone if c.isdigit())
    link = f"https://web.whatsapp.com/send?phone={numero}&text={quote(mensagem)}"
    if len(link) > 4000:
        return "Mensagem longa demais para o WhatsApp. Envie uma versão mais curta."
    webbrowser.open(link)
    return ("WhatsApp Web aberto com a mensagem pronta. "
            f"Falta o usuário clicar em Enviar. Link: {link[:80]}...")


# Sem SMTP configurado no .env, o e-mail é apenas salvo em ./saida (simulação).
@tool
def enviar_email(destinatario: str, assunto: str, corpo: str) -> str:
    """Envia um e-mail. Use somente depois que o resumo estiver pronto."""
    if os.getenv("SMTP_HOST"):
        mensagem = EmailMessage()
        mensagem["From"] = os.environ["SMTP_USUARIO"]
        mensagem["To"] = destinatario
        mensagem["Subject"] = assunto
        mensagem.set_content(corpo)
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORTA", "587"))) as smtp:
            smtp.starttls()
            smtp.login(os.environ["SMTP_USUARIO"], os.environ["SMTP_SENHA"])
            smtp.send_message(mensagem)
        return f"E-mail enviado para {destinatario}."

    pasta = PASTA / "saida"
    pasta.mkdir(exist_ok=True)
    arquivo = pasta / f"email_{datetime.now():%Y%m%d_%H%M%S}.txt"
    arquivo.write_text(f"Para: {destinatario}\nAssunto: {assunto}\n\n{corpo}\n", encoding="utf-8")
    return f"[simulação] E-mail para {destinatario} salvo em {arquivo.name}."


# ---------------------------------------------------------------------------
# ETAPA 6 — Memória de curto prazo + montagem do agente
# ---------------------------------------------------------------------------
def criar_agente(llm=None):
    return create_agent(
        model=llm or criar_llm(),
        tools=[buscar_artigos, buscar_contato, enviar_whatsapp, enviar_email],
        system_prompt=INSTRUCOES,
        checkpointer=InMemorySaver(),          # guarda o histórico de cada conversa
        middleware=[
            # Limite de autonomia: enviar e-mail exige aprovação humana.
            # (No WhatsApp, a aprovação é o próprio clique em "Enviar".)
            HumanInTheLoopMiddleware(interrupt_on={"enviar_email": True}),
        ],
    )


# ---------------------------------------------------------------------------
# ETAPA 7 — Execução do ciclo percepção → raciocínio → ação
# ---------------------------------------------------------------------------
def mostrar_passos(mensagens, inicio=0):
    """Mostra o que aconteceu em cada volta do ciclo (fluxo de execução)."""
    for msg in mensagens[inicio:]:
        tipo = msg.type
        if tipo == "ai" and msg.tool_calls:
            for chamada in msg.tool_calls:
                print(f"  [decisão]    {chamada['name']}({json.dumps(chamada['args'], ensure_ascii=False)[:120]})")
        elif tipo == "tool":
            print(f"  [observação] {msg.name}: {str(msg.content)[:120]}...")


def conversar(agente, texto, conversa="aluno-01", aprovar=None):
    """Envia um pedido ao agente e trata a aprovação humana, se houver."""
    config = {"configurable": {"thread_id": conversa}}
    antes = len(agente.get_state(config).values.get("messages", []))
    resultado = agente.invoke({"messages": [{"role": "user", "content": texto}]}, config)

    while "__interrupt__" in resultado:
        pedido = resultado["__interrupt__"][0].value
        decisoes = []
        for acao in pedido["action_requests"]:
            args = acao["args"]
            print("\n--- Aprovação necessária -----------------------------")
            print(f"Para:    {args.get('destinatario')}")
            print(f"Assunto: {args.get('assunto')}")
            print(f"{args.get('corpo')}")
            print("------------------------------------------------------")
            ok = aprovar(acao) if aprovar else input("Enviar este e-mail? [s/n] ").strip().lower() == "s"
            decisoes.append({"type": "approve"} if ok else
                            {"type": "reject", "message": "O usuário não aprovou o envio."})
        resultado = agente.invoke(Command(resume={"decisions": decisoes}), config)

    mostrar_passos(resultado["messages"], antes)
    return resultado["messages"][-1].content


# ---------------------------------------------------------------------------
# ETAPA 8 — Demonstração
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agente = criar_agente()
    print(f"Agente pronto (modelo: {MODELO}). Digite 'sair' para encerrar.\n")
    pedido = "Resuma os 3 artigos mais recentes sobre RAG e envie para minha orientadora."
    print(f"Você: {pedido}")
    print(f"\nAgente: {conversar(agente, pedido)}\n")

    while True:
        texto = input("Você: ").strip()
        if texto.lower() in {"sair", "exit", "quit"}:
            break
        if texto:
            print(f"\nAgente: {conversar(agente, texto)}\n")
