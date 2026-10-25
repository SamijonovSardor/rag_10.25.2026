"""Retrieval -> kontekst bilan prompt -> generatsiya -> manba tekshirish."""
import re
import time

from .llm import OpenRouterLLM
from .retrieval import Retriever


class RAG:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = OpenRouterLLM()

    def ask(self, question, language="uz", top_k=5):
        start = time.monotonic()
        context = self.retriever.search(question, top_k)
        generated = self.llm.answer(question, context, language)
        allowed = {item["reference"] for item in context}
        cited = generated.get("sources", [])
        inline = {int(x) for x in re.findall(r"\[(\d+)\]", generated.get("answer", ""))}
        valid = (generated.get("supported") is True and bool(cited)
                 and all(type(x) is int and x in allowed for x in cited)
                 and inline <= set(cited))
        if not valid:
            answer = ("Bu ma'lumot qo'llanmada topilmadi." if language == "uz"
                      else "This information was not found in the manual.")
            sources = []
        else:
            answer = generated["answer"].strip()
            if not inline:
                answer += " " + " ".join(f"[{i}]" for i in dict.fromkeys(cited))
            sources = [x for x in context if x["reference"] in cited]
        return {"question": question, "answer": answer, "supported": valid,
                "sources": sources, "retrieved": context,
                "elapsed_seconds": round(time.monotonic() - start, 2)}

    def close(self):
        self.llm.close()
        self.retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
