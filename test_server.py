#!/usr/bin/env python3
"""
Simple test script for Veridika server.
Tests basic connectivity and functionality.

Usage:
    python test_server.py
    python test_server.py --url <url> --api-key <api_key>
"""

import argparse
import asyncio
import time

import httpx


async def test_health(base_url: str):
    """Test health endpoint (no auth required)."""
    print("🔍 Testing health endpoint...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base_url}/health")
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "healthy":
                print("✅ Health check passed")
                print(f"   Server ID: {data.get('server_id', 'Unknown')}")
                return True
            else:
                print(f"❌ Health check failed: {data}")
                return False

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def test_status(base_url: str, api_key: str):
    """Test status endpoint (auth required)."""
    print("\n🔍 Testing status endpoint...")

    try:
        headers = {"X-API-Key": api_key}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base_url}/status", headers=headers)
            response.raise_for_status()
            data = response.json()

            print("✅ Status check passed")
            print(f"   Server ID: {data.get('server_id', 'Unknown')}")
            print(f"   Processing: {data.get('is_processing', 'Unknown')}")
            print(
                f"   Active Jobs: {data.get('active_jobs_count', 'Unknown')} / {data.get('max_concurrent_jobs', 'Unknown')}"
            )
            return True

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("❌ Authentication failed (401) - Missing API key")
        elif e.response.status_code == 403:
            print("❌ Authentication failed (403) - Invalid API key")
        else:
            print(f"❌ Status check failed: HTTP {e.response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False


async def test_generate_image(base_url: str, api_key: str, timeout_seconds: int = 120):
    """Test image generation endpoint."""
    print("\n🔍 Testing generate-image endpoint...")
    try:
        headers = {"X-API-Key": api_key}
        body = {
            "input": "Coches eléctricos en una ciudad moderna",
            "model": "google/gemini-2.5-flash",
            "size": "1280x256",
        }
        async with httpx.AsyncClient(timeout=timeout_seconds + 10) as client:
            submit = await client.post(
                f"{base_url}/stepwise/generate-image", json=body, headers=headers
            )
            submit.raise_for_status()
            job_id = submit.json().get("job_id")
            if not job_id:
                print("❌ No job ID returned for generate-image")
                return False
            print(f"   🖼️ generate-image job: {job_id}")
            result = await poll_job_result(client, base_url, job_id, headers, timeout_seconds)
            payload = result.get("result", {})
            if payload.get("image_description"):
                print("   ✅ generate-image completed")
                return True
            print(f"   ❌ generate-image returned empty or invalid result: {payload}")
            return False
    except Exception as e:
        print(f"❌ generate-image test failed: {e}")
        return False


async def poll_job_result(
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
    headers: dict,
    timeout_seconds: int = 120,
    poll_interval: float = 2.0,
):
    """Poll /result/{job_id} until completed or timeout. Returns the result JSON when done."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        result_response = await client.get(f"{base_url}/result/{job_id}", headers=headers)

        if result_response.status_code == 404:
            await asyncio.sleep(poll_interval)
            continue

        result_response.raise_for_status()
        result_data = result_response.json()
        status = result_data.get("status")
        if status == "completed":
            return result_data
        if status == "in_progress":
            await asyncio.sleep(poll_interval)
            continue
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Timeout waiting for job {job_id}")


async def test_fact_check(base_url: str, api_key: str, timeout_seconds: int = 120):
    """Test fact-checking endpoint."""
    print("\n🔍 Testing fact-check endpoint...")

    try:
        job_data = {
            "article_topic": "Los coches eléctricos son más contaminantes que los coches de gasolina",
            "language": "es",
            "location": "es",
            "config": "pro",
        }

        headers = {"X-API-Key": api_key}

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Submit job
            response = await client.post(f"{base_url}/fact-check", json=job_data, headers=headers)
            response.raise_for_status()
            result = response.json()
            job_id = result.get("job_id")

            if not job_id:
                print("❌ No job ID returned")
                return False

            print(f"✅ Job submitted successfully: {job_id}")

            # Wait for result
            print(f"   Waiting for result (timeout {timeout_seconds}s)...")
            result_data = await poll_job_result(client, base_url, job_id, headers, timeout_seconds)
            print("✅ Job completed successfully")
            if result_data.get("result"):
                job_result = result_data["result"]
                if job_result.get("success", True):
                    print("   ✅ Fact-checking succeeded")
                    print(f"   Answer preview: {job_result.get('answer', 'N/A')[:100]}...")
                else:
                    print(f"   ❌ Fact-checking failed: {job_result.get('error', 'Unknown error')}")
            return True

    except httpx.HTTPStatusError as e:
        print(f"❌ Fact-check test failed: HTTP {e.response.status_code}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Fact-check test failed: {e}")
        return False


async def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test Veridika server")
    parser.add_argument("--url", default=None, help="Server URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--timeout", type=int, default=120, help="Per-job timeout seconds")
    parser.add_argument("--quick", action="store_true", help="Only health and status (fast)")

    args = parser.parse_args()

    print("🚀 Veridika Server Test")
    print(f"   Server: {args.url}")
    print(f"   API Key: {args.api_key[:8]}...{args.api_key[-4:]}")
    print("=" * 50)

    tests_passed = 0
    total_tests = 2 if args.quick else 7

    # Test 1: Health check
    if await test_health(args.url):
        tests_passed += 1

    # Test 2: Status check
    if await test_status(args.url, args.api_key):
        tests_passed += 1

    if not args.quick:
        # Test 3: Fact-check (queued)
        if await test_fact_check(args.url, args.api_key, args.timeout):
            tests_passed += 1

        # Test 4-6: Stepwise flow
        stepwise_ok = await test_stepwise_flow(args.url, args.api_key, args.timeout)
        tests_passed += 3 if stepwise_ok else 0

        # Test 7: Generate image
        if await test_generate_image(args.url, args.api_key, args.timeout):
            tests_passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Server is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check server configuration.")
        return 1


# ===================== Additional endpoint tests =====================


async def test_stepwise_flow(base_url: str, api_key: str, timeout_seconds: int = 180):
    """Run the full stepwise flow: questions -> refine -> search -> refine-sources -> generate-article.
    Returns True if at least the last step succeeds; prints progress for each stage.
    """
    print("\n🔍 Testing stepwise workflow endpoints...")
    headers = {"X-API-Key": api_key}
    model = "Latxa8B"  # "google/gemini-2.5-flash" #Latxa70B
    input_text = "Los coches eléctricos son más contaminantes que los coches de gasolina"
    language = "es"
    location = "es"

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds + 10) as client:
            # 1) critical-questions
            cq_body = {
                "input": input_text,
                "model": model,
                "language": language,
                "location": location,
            }
            cq_resp = await client.post(
                f"{base_url}/stepwise/critical-questions", json=cq_body, headers=headers
            )
            cq_resp.raise_for_status()
            cq_job_id = cq_resp.json().get("job_id")
            print(f"   📝 critical-questions job: {cq_job_id}")
            cq_result = await poll_job_result(client, base_url, cq_job_id, headers, timeout_seconds)
            questions = cq_result.get("result", {}).get("questions", [])
            cost1 = cq_result.get("result", {}).get("cost")
            print(f"   ✅ critical-questions completed: {len(questions)} questions, cost={cost1}")
            if not questions:
                print("   ❌ No questions returned, aborting stepwise test")
                return False

            # 2) refine-questions
            rq_body = {
                "questions": questions,
                "input": input_text,
                "refinement": "Por favor, considera el reciclaje de baterías",
                "language": language,
                "location": location,
                "model": model,
            }
            rq_resp = await client.post(
                f"{base_url}/stepwise/refine-questions", json=rq_body, headers=headers
            )
            rq_resp.raise_for_status()
            rq_job_id = rq_resp.json().get("job_id")
            print(f"   📝 refine-questions job: {rq_job_id}")
            rq_result = await poll_job_result(client, base_url, rq_job_id, headers, timeout_seconds)
            questions_refined = rq_result.get("result", {}).get("questions", [])
            cost2 = rq_result.get("result", {}).get("cost")
            print(
                f"   ✅ refine-questions completed: {len(questions_refined)} questions, cost={cost2}"
            )
            if not questions_refined:
                print("   ❌ No refined questions returned, aborting stepwise test")
                return False

            # 3) search-sources
            ss_body = {
                "questions": questions_refined,
                "input": input_text,
                "language": language,
                "location": location,
                "model": model,
                "web_search_provider": "serper",
            }
            ss_resp = await client.post(
                f"{base_url}/stepwise/search-sources", json=ss_body, headers=headers
            )
            ss_resp.raise_for_status()
            ss_job_id = ss_resp.json().get("job_id")
            print(f"   🔎 search-sources job: {ss_job_id}")
            ss_result = await poll_job_result(client, base_url, ss_job_id, headers, timeout_seconds)
            sources = ss_result.get("result", {}).get("sources", [])
            searches = ss_result.get("result", {}).get("searches", [])
            cost3 = ss_result.get("result", {}).get("cost")
            print(
                f"   ✅ search-sources completed: {len(sources)} sources, {len(searches)} searches, cost={cost3}"
            )
            if not sources:
                print("   ❌ No sources returned, aborting stepwise test")
                return False

            # 4) refine-sources
            rs_body = {
                "input": input_text,
                "searches": searches,
                "language": language,
                "location": location,
                "refinement": "Prefiero fuentes oficiales de la UE",
                "model": model,
                "web_search_provider": "serper",
            }
            rs_resp = await client.post(
                f"{base_url}/stepwise/refine-sources", json=rs_body, headers=headers
            )
            rs_resp.raise_for_status()
            rs_job_id = rs_resp.json().get("job_id")
            print(f"   🔎 refine-sources job: {rs_job_id}")
            rs_result = await poll_job_result(client, base_url, rs_job_id, headers, timeout_seconds)
            sources_refined = rs_result.get("result", {}).get("sources", [])
            searches_refined = rs_result.get("result", {}).get("searches", [])
            cost4 = rs_result.get("result", {}).get("cost")
            print(
                f"   ✅ refine-sources completed: {len(sources_refined)} sources, {len(searches_refined)} searches, cost={cost4}"
            )
            if not sources_refined:
                print("   ❌ No refined sources returned, aborting stepwise test")
                return False

            # 5) generate-article
            ga_body = {
                "questions": questions_refined,
                "input": input_text,
                "language": language,
                "location": location,
                "sources": sources_refined,
                "model": model,
                "use_rag": True,
                "embedding_model": "gemini-embedding-001",
            }
            ga_resp = await client.post(
                f"{base_url}/stepwise/generate-article", json=ga_body, headers=headers
            )
            ga_resp.raise_for_status()
            ga_job_id = ga_resp.json().get("job_id")
            print(f"   📰 generate-article job: {ga_job_id}")
            ga_result = await poll_job_result(client, base_url, ga_job_id, headers, timeout_seconds)
            article = ga_result.get("result", {})
            cost5 = article.get("cost")
            if article:
                print(f"   ✅ generate-article completed, cost={cost5}")
                return True
            print("   ❌ generate-article returned empty result")
            return False
    except TimeoutError as e:
        print(f"❌ Stepwise test timed out: {e}")
        return False
    except Exception as e:
        print(f"❌ Stepwise test failed: {e}")
        return False


if __name__ == "__main__":
    try:
        import httpx
    except ImportError:
        print("❌ Please install httpx: pip install httpx")
        exit(1)

    exit(asyncio.run(main()))
