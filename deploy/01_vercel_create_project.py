import getpass
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

team_id = input("Vercel Team ID: ").strip()
token = getpass.getpass("Vercel Access Token (hidden input): ").strip()

if not team_id.startswith("team_") or not token:
    raise SystemExit("Enter a Team ID starting with team_ and a valid token.")

url = "https://api.vercel.com/v11/projects?" + urlencode(
    {
        "teamId": team_id,
    }
)

request = Request(
    url,
    data=json.dumps({"name": "model-anatomy-demo"}).encode(),
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urlopen(request, timeout=30) as response:
        project = json.load(response)
except HTTPError as error:
    hints = {
        401: "Check whether the token is valid and has not expired.",
        403: "Check whether the token scope includes this team.",
        409: "A project with this name may already exist. Check the dashboard.",
    }
    print(f"Project creation failed: HTTP {error.code}")
    print(hints.get(error.code, "Record the status code before continuing to the next step."))
    raise SystemExit(1)
except (URLError, TimeoutError):
    raise SystemExit("The network request failed. Check the connection to api.vercel.com.")

print("\nProject created successfully")
print("Project name:", project["name"])
print("Project ID:", project["id"])
print("Team ID:", team_id)
