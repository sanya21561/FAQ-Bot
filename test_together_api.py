#!/usr/bin/env python3
"""
Test script for Together API integration
"""

from models.together_inference import query_together_llm, query_together_llm_with_system_prompt

def test_basic_query():
    """Test basic query functionality"""
    print("Testing basic query...")
    try:
        response = query_together_llm("What is Jupiter Money?")
        print(f"✅ Basic query successful!")
        print(f"Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Basic query failed: {e}")
        return False

def test_system_prompt():
    """Test system prompt functionality"""
    print("\nTesting system prompt...")
    try:
        system_prompt = "You are a helpful assistant that answers questions about Jupiter Money."
        user_prompt = "What services does Jupiter offer?"
        response = query_together_llm_with_system_prompt(system_prompt, user_prompt)
        print(f"✅ System prompt query successful!")
        print(f"Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ System prompt query failed: {e}")
        return False

def test_rag_style_prompt():
    """Test RAG-style prompt that matches our use case"""
    print("\nTesting RAG-style prompt...")
    try:
        prompt = (
            "User question: What is Jupiter Money?\n\n"
            "Relevant FAQ:\nQ: What is Jupiter Money?\nA: Jupiter Money is a digital banking platform that offers various financial services.\n\n"
            "Respond conversationally. If the user greets, greet back and ask if they have questions about Jupiter Money. "
            "If the FAQ is relevant, answer using its info. If not, say you don't know. "
            "Start your reply with 'FINAL ANSWER:' and output only the answer."
        )
        response = query_together_llm(prompt)
        print(f"✅ RAG-style prompt successful!")
        print(f"Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ RAG-style prompt failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Together API Integration")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    if test_basic_query():
        success_count += 1
    
    if test_system_prompt():
        success_count += 1
    
    if test_rag_style_prompt():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Together API integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.") 