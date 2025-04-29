import requests
from bs4 import BeautifulSoup
import os

def scrape_team_logos():
    url = "https://www.mlb.com/team"  # Update to the correct MLB team page URL
    print(":) Scraping MLB team logos...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Create a directory to save logos
        os.makedirs("logos", exist_ok=True)

        # Adjust selector to match the new structure of the logo elements on the MLB team page
        logo_elements = soup.find_all("img", class_="p-forge-logo")
        for logo in logo_elements:
            team_name = logo.get("alt", "unknown_team").replace(" logo", "").replace(" ", "_").lower()
            logo_url = logo.get("data-src") or logo.get("src")  # Use data-src if available, fallback to src

            if logo_url and team_name:
                # Ensure the logo URL is complete
                if logo_url.startswith("//"):
                    logo_url = "https:" + logo_url

                # Download and save the logo
                logo_response = requests.get(logo_url)
                logo_path = os.path.join("logos", f"{team_name}.svg")
                with open(logo_path, "wb") as file:
                    file.write(logo_response.content)
                print(f":) Saved logo for {team_name} at {logo_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error during scraping: {e}")

if __name__ == "__main__":
    scrape_team_logos()