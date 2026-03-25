import asyncio
import threading
import json


async def make_sse_stream(pipeline, question: str):
    """
    Async generator yielding SSE-formatted strings.
    """
    # ── Retrieval ─────────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    try:
        safe_contexts, results = await loop.run_in_executor(
            None, pipeline._retrieve_context, question
        )
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        return

    # ── Send sources as first event ───────────────────────────────────────
    import os
    sources = []
    for r in (results or [])[:6]:
        src = r.get("source", "")
        sources.append({
            "text":       (r.get("text") or "")[:220],
            "source":     os.path.basename(src),
            "modality":   r.get("modality", "text"),
            "score":      round(float(r.get("score") or 0), 3),
            "section":    r.get("section", ""),
            "page":       r.get("page"),
            "start_time": r.get("start_time"),
        })
    yield f"data: {json.dumps({'sources': sources})}\n\n"

    # ── Empty context ─────────────────────────────────────────────────────
    if not safe_contexts:
        yield f"data: {json.dumps({'token': 'I could not find relevant information.'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    # ── Stream tokens ─────────────────────────────────────────────────────
    from src.generation.prompt_templates import build_prompt
    prompt = build_prompt(safe_contexts, question)
    pipeline._ensure_generator()

    queue = asyncio.Queue()

    def _worker():
        try:
            for token in pipeline.generator.generate_stream(prompt):
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        event_type, data = await queue.get()
        if event_type == "token":
            yield f"data: {json.dumps({'token': data})}\n\n"
        elif event_type == "error":
            yield f"data: {json.dumps({'error': data})}\n\n"
            break
        elif event_type == "done":
            yield f"data: {json.dumps({'done': True})}\n\n"
            break