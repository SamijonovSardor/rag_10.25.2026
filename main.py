"""Honda RAG + OpenRouter. Tayyorlash: python main.py setup"""
import argparse
import json
import sys


def print_sources(sources, show_text=False):
    from honda_rag import config as c
    for item in sources:
        book = f", kitob sahifasi {item['book_page']}" if item.get("book_page") is not None else ""
        print(f"  [{item['reference']}] {item['title']} | PDF sahifasi {item['pdf_page']}{book}")
        print(f"      {c.PDF_URL}#page={item['pdf_page']}")
        if show_text:
            print(f"      Qidiruv: {item.get('search_query', '')}")
            print(f"      BM25 so'rovi: {item.get('lexical_query', '')}")
            print(f"      cosine={item['semantic_score']:.4f}; BM25={item['bm25_score']:.3f}")
            print("      " + item["text"])


def print_answer(result, show_context=False):
    print("\nJavob:")
    print(result["answer"])
    if result["sources"]:
        print("\nManbalar:")
        print_sources(result["sources"])
    if show_context:
        print("\nQidiruvda topilgan kontekst:")
        print_sources(result["retrieved"], show_text=True)
    print(f"\nVaqt: {result['elapsed_seconds']:.1f} soniya\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Honda qo'llanmasi asosida OpenRouter RAG (terminal).")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("setup", help="PDFni tayyorlash va OpenRouter embedding indeksini yaratish")
    commands.add_parser("check", help="Model nomlarini ommaviy OpenRouter katalogida tekshirish")
    commands.add_parser("auth", help="API kaliti haqiqiyligini va oddiy inference kaliti ekanini tekshirish")
    ingest = commands.add_parser("ingest", help="PDFni qayta indekslash")
    ingest.add_argument("--force", action="store_true")
    commands.add_parser("info", help="Kitob va indeks haqida")
    for name in ("ask", "chat", "search"):
        sub = commands.add_parser(name)
        if name != "chat":
            sub.add_argument("question")
        sub.add_argument("--top-k", type=int, choices=range(1, 9), default=5)
        sub.add_argument("--language", choices=("uz", "en"), default="uz")
        sub.add_argument("--show-context", action="store_true")
        if name == "ask":
            sub.add_argument("--json", action="store_true", help="Natijani JSON shaklida chiqarish")
    args = parser.parse_args(argv)
    command = args.command or "chat"
    if command == "setup":
        from honda_rag.setup import setup
        setup()
        print("\nTayyor! Suhbatni boshlash: python main.py")
    elif command == "ingest":
        from honda_rag.retrieval import build_index
        build_index(force=args.force)
    elif command == "check":
        from honda_rag.setup import check_models
        check_models()
    elif command == "auth":
        from honda_rag.openrouter import OpenRouterClient
        client = OpenRouterClient()
        try:
            client.validate_api_key()
            print("OpenRouter kaliti tanildi: oddiy inference API kaliti. Kalit qiymati chiqarilmadi.")
        finally:
            client.close()
    elif command == "info":
        from honda_rag import config as c
        path = c.INDEX / "metadata.json"
        try:
            c.require_api_key()
            has_key = True
        except RuntimeError:
            has_key = False
        info = {"backend": "openrouter", "chat_model": c.CHAT_MODEL,
                "embedding_model": c.EMBEDDING_MODEL, "api_key_configured": has_key,
                "index_directory": str(c.INDEX), "index_exists": path.exists()}
        if path.exists():
            info["index_metadata"] = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(info, ensure_ascii=False, indent=2))
    elif command == "search":
        from honda_rag.retrieval import Retriever
        with Retriever() as retriever:
            print_sources(retriever.search(args.question, args.top_k), show_text=True)
    else:
        from honda_rag.pipeline import RAG
        with RAG() as rag:
            if command == "ask":
                if args.json:
                    # Diagnostik xabarlar JSON stdout oqimiga aralashmaydi.
                    from contextlib import redirect_stdout
                    with redirect_stdout(sys.stderr):
                        result = rag.ask(args.question, args.language, args.top_k)
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print_answer(rag.ask(args.question, args.language, args.top_k), args.show_context)
            else:
                language = getattr(args, "language", "uz")
                show_context = getattr(args, "show_context", False)
                top_k = getattr(args, "top_k", 5)
                from honda_rag import config as c
                print("\nHONDA RAG | 2023 Civic Sedan Owner's Manual | OpenRouter")
                print(f"Javob: {c.CHAT_MODEL}; embedding: {c.EMBEDDING_MODEL}")
                print("Savollar o'zbekcha yoki inglizcha. Har savol mustaqil ko'rib chiqiladi.")
                print("Buyruqlar: /exit, /help, /sources, /language uz, /language en\n")
                while True:
                    try:
                        question = input("Siz > ").strip()
                    except EOFError:
                        break
                    if question.lower() in ("/exit", "exit", "quit", "chiqish"):
                        break
                    if not question:
                        continue
                    if question == "/help":
                        print("Masalan: Qanday motor moyi tavsiya etiladi?\n"
                              "Masalan: What fuel octane rating is recommended?\n"
                              "/sources - topilgan matnni ko'rsatish/o'chirish; /exit - chiqish.\n")
                        continue
                    if question == "/sources":
                        show_context = not show_context
                        print(f"Kontekst ko'rsatish: {show_context}\n")
                        continue
                    if question in ("/language uz", "/language en"):
                        language = question.split()[1]
                        print(f"Javob tili: {language}\n")
                        continue
                    try:
                        print_answer(rag.ask(question, language, top_k), show_context)
                    except (ValueError, RuntimeError) as exc:
                        print(f"Xato: {exc}\n")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except KeyboardInterrupt:
        print("\nSuhbat tugadi.")
    except Exception as exc:
        print(f"Xato: {exc}", file=sys.stderr)
        sys.exit(1)
