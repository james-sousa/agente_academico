# Agente de pesquisa acadêmica

Exemplo prático do minicurso **Introdução aos Agentes de IA (XV SINFO)**, feito com LangChain e a API do Hugging Face.

O agente executa o pedido:

> Resuma os 3 artigos mais recentes sobre RAG e envie para minha orientadora.

## Arquivos

| Arquivo | Para quê |
|---|---|
| `agente_pesquisa.ipynb` | Notebook para a demonstração ao vivo, uma célula por etapa |
| `agente.py` | O mesmo agente em um script único (`python agente.py`) |
| `contatos.json` | Base de contatos (memória de longo prazo): telefone, e-mail e canal preferido |
| `.env.example` | Modelo de configuração: token do Hugging Face e SMTP opcional |

## Como rodar

```bash
pip install -r requirements.txt
cp .env.example .env      # edite e coloque seu HF_TOKEN
python agente.py          # ou abra o notebook
```

O token é criado em huggingface.co/settings/tokens e precisa da permissão **Make calls to Inference Providers**.

## Componentes → código

| Componente | Onde está |
|---|---|
| LLM | `criar_llm()`: `HuggingFaceEndpoint` + `ChatHuggingFace` |
| Instruções e objetivo | `INSTRUCOES` (prompt de sistema) |
| Ferramenta com API externa | `buscar_artigos` (API do arXiv, sem chave) |
| Base de dados | `buscar_contato` (lê `contatos.json`) |
| Ferramentas de ação | `enviar_whatsapp` (WhatsApp Web, sem API) e `enviar_email` (simulado por padrão) |
| Memória de curto prazo | `checkpointer=InMemorySaver()` + `thread_id` |
| Limite de autonomia | WhatsApp: você clica em Enviar · e-mail: `HumanInTheLoopMiddleware` pede aprovação |
| Ciclo e fluxo de execução | `create_agent(...)` e `conversar()` / `mostrar_passos()` |

## Observações

- **Modelo:** o padrão é `Qwen/Qwen2.5-72B-Instruct`. Qualquer modelo de chat com suporte a *tool calling* nos Inference Providers funciona; troque em `HF_MODELO`. Modelos pequenos erram mais ao escolher ferramentas.
- **WhatsApp sem API:** a ferramenta abre `web.whatsapp.com/send?phone=...&text=...` no navegador com a mensagem pronta; o envio é o clique em **Enviar**. Deixe o WhatsApp Web conectado antes da aula. Não há automação de cliques, o que evita violar os termos do WhatsApp. O agente usa o `canal_preferido` do contato quando o pedido não diz o canal.
- **E-mail:** sem as variáveis `SMTP_*`, nada é enviado de verdade. O e-mail aprovado é salvo em `./saida`.
- **Testes:** o fluxo completo foi testado com um LLM simulado, incluindo aprovação, rejeição e memória. As chamadas reais ao Hugging Face e ao arXiv não foram testadas daqui, então rode uma vez antes da aula.
