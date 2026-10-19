"""Run with YOUR credentials in .env:   python scripts/verify_providers.py
Makes REAL requests (read-only: quotes/history/LLM test prompts). Never places an order. Writes
outputs/provider_verification.json (no secrets) and prints PASS / FAIL / BLOCKED / NOT_CONFIGURED per provider."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import provider_verification as pv

if __name__ == "__main__":
    recs = pv.run_all()
    print(f"{'provider':11s} {'endpoint class':26s} {'instrument':14s} outcome")
    for r in recs:
        print(f"{r['provider']:11s} {r['endpoint_class']:26s} {str(r['instrument'] or '-'):14s} {r['outcome']}"
              + (f"   ({r['detail']})" if r["detail"] else ""))
    print("\nSaved to", pv.save(recs))
    sys.exit(1 if any(r["outcome"] == "FAIL" for r in recs) else 0)
