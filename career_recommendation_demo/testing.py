import urllib.request
import json

# Your API Key from the screenshot
API_KEY = "864tf-QuKjE-zseDp-LWbRI"

# We will test the 'About' endpoint first (Basic check)
# Then the 'Search' endpoint (Service check)
test_endpoints = [
    "https://api-v2.onetcenter.org/about/",
    "https://api-v2.onetcenter.org/online/search?keyword=engineer"
]

headers = {
    "X-API-Key": API_KEY,
    "Accept": "application/json",
    "User-Agent": "python-OnetWebService/2.00"
}

for url in test_endpoints:
    print(f"\nTesting: {url}")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            print("✅ SUCCESS! Data received.")
            # Print first job title if it's the search endpoint
            if "occupation" in res_data:
                print(f"Sample Job Found: {res_data['occupation'][0]['title']}")
    except Exception as e:
        print(f"❌ FAILED: {e}")