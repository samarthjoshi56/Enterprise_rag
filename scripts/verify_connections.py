#!/usr/bin/env python3
"""CLI script to verify PostgreSQL, Qdrant, and Redis database connections."""

import asyncio
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.postgres import check_postgres_connection
from app.db.qdrant import check_qdrant_connection
from app.db.redis import check_redis_connection
from app.api.health import verify_framework_imports


async def main():
    print("=" * 60)
    print(" Enterprise RAG - System Connections & Verification")
    print("=" * 60)

    # Check Framework Imports
    print("\n1. Verifying Framework Imports:")
    frameworks = verify_framework_imports()
    for fw_name, info in frameworks.items():
        if info["status"] == "available":
            print(f"   [OK] {fw_name:<18} (v{info['version']})")
        else:
            print(f"   [FAIL] {fw_name:<16}: {info.get('error')}")

    # Check Database Connections
    print("\n2. Checking Infrastructure Services:")

    pg_info = await check_postgres_connection()
    if pg_info["status"] == "connected":
        print(f"   [OK] PostgreSQL : Connected (Latency: {pg_info['latency_ms']} ms | Version: {pg_info.get('version')})")
    else:
        print(f"   [FAIL] PostgreSQL : Error - {pg_info.get('error')}")

    qdrant_info = await check_qdrant_connection()
    if qdrant_info["status"] == "connected":
        print(f"   [OK] Qdrant     : Connected (Latency: {qdrant_info['latency_ms']} ms | Collections: {qdrant_info.get('collections_count')})")
    else:
        print(f"   [FAIL] Qdrant     : Error - {qdrant_info.get('error')}")

    redis_info = await check_redis_connection()
    if redis_info["status"] == "connected":
        print(f"   [OK] Redis      : Connected (Latency: {redis_info['latency_ms']} ms | Version: {redis_info.get('version')})")
    else:
        print(f"   [FAIL] Redis      : Error - {redis_info.get('error')}")

    print("=" * 60)

    all_connected = (
        pg_info["status"] == "connected"
        and qdrant_info["status"] == "connected"
        and redis_info["status"] == "connected"
    )

    if all_connected:
        print("  SUCCESS: All 3 infrastructure services (PostgreSQL, Qdrant, Redis) connected!")
        sys.exit(0)
    else:
        print("  WARNING: One or more services failed to connect. Ensure services are running.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
