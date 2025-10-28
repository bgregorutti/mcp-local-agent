#!/usr/bin/env python3
"""
Test script to verify local model backend connection.
Run this before setting up the MCP server to ensure everything works.
"""

import urllib.request
import json
import sys


def test_lmstudio(base_url="http://localhost:1234/v1"):
    """Test LM Studio connection"""
    print(f"🔍 Testing LM Studio connection at {base_url}...")

    try:
        # Test 1: Check if server is running
        req = urllib.request.Request(f"{base_url}/models")
        with urllib.request.urlopen(req, timeout=5) as response:
            models = json.loads(response.read().decode('utf-8'))
            print("✅ LM Studio server is running")

            if models.get('data'):
                print(f"✅ Found {len(models['data'])} model(s) loaded:")
                for model in models['data']:
                    print(f"   - {model['id']}")
            else:
                print("⚠️  No models loaded in LM Studio")
                print("   → Please load a model in LM Studio before continuing")
                return False

        # Test 2: Try a simple completion
        if models.get('data'):
            # Skip embedding models
            chat_models = [m for m in models['data'] if 'embed' not in m['id'].lower()]
            if not chat_models:
                print("⚠️  Only embedding models found. Load a chat/completion model.")
                return False

            test_model = chat_models[0]['id']
            print(f"\n🧪 Testing completion with model: {test_model}")

            request_data = {
                "model": test_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello from local model!' and nothing else."}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }

            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(request_data).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                reply = result['choices'][0]['message']['content']
                print(f"✅ Model response: {reply[:100]}")
                print("✅ LM Studio is working correctly!")
                return True

    except urllib.error.URLError as e:
        print(f"❌ Connection failed: {e}")
        print("\n📋 Troubleshooting steps:")
        print("   1. Make sure LM Studio is running")
        print("   2. Load a model in LM Studio")
        print("   3. Start the local server (Server tab → Start Server)")
        print(f"   4. Verify URL: {base_url}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_ollama(base_url="http://localhost:11434"):
    """Test Ollama connection"""
    print(f"🔍 Testing Ollama connection at {base_url}...")

    try:
        # Check if Ollama is running
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = data.get('models', [])

            print("✅ Ollama server is running")

            if models:
                print(f"✅ Found {len(models)} model(s):")
                for model in models:
                    print(f"   - {model['name']}")
            else:
                print("⚠️  No models installed")
                print("   → Run: ollama pull deepseek-coder:6.7b")
                return False

            return True

    except urllib.error.URLError as e:
        print(f"❌ Connection failed: {e}")
        print("\n📋 Troubleshooting steps:")
        print("   1. Install Ollama: https://ollama.com/download")
        print("   2. Run: ollama serve")
        print("   3. Pull a model: ollama pull deepseek-coder:6.7b")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    print("🚀 Local Agent MCP Server - Connection Test\n")
    print("=" * 60)

    # Test LM Studio (default)
    print("\n[1/2] Testing LM Studio (default backend)")
    print("-" * 60)
    lmstudio_ok = test_lmstudio()

    print("\n")
    print("=" * 60)

    # Test Ollama (optional)
    print("\n[2/2] Testing Ollama (optional backend)")
    print("-" * 60)
    ollama_ok = test_ollama()

    print("\n")
    print("=" * 60)
    print("\n📊 Summary:")
    print("-" * 60)

    if lmstudio_ok:
        print("✅ LM Studio: Ready to use")
    else:
        print("❌ LM Studio: Not ready")

    if ollama_ok:
        print("✅ Ollama: Ready to use")
    else:
        print("⚠️  Ollama: Not available (optional)")

    print("\n")

    if lmstudio_ok or ollama_ok:
        print("🎉 Success! At least one backend is ready.")
        print("\n📝 Next steps:")
        print("   1. Add MCP server to Claude Code config")
        print("   2. Restart Claude Code")
        print("   3. Start using delegation tools!")
        print("\nSee README.md for configuration details.")
        return 0
    else:
        print("❌ No backends available. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
