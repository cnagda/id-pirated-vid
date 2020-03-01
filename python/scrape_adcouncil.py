#!/usr/bin/env python3

import requests
import os
from bs4 import BeautifulSoup
from shutil import rmtree
from sys import argv
import argparse

# List of campaigns
BROWSE_URL = "https://www.psacentral.org/browse-campaigns"

# Base URL for requesting campaign JSON objects
API_URL = "https://www.psacentral.org/api/group?id="


class Campaign:
    def __init__(self, id, name):
        self.id = id
        self.name = name


class Asset:
    def __init__(self, campaign, url, title, length, file_format):
        self.campaign = campaign
        self.url = url
        self.title = title
        self.length = length
        self.file_format = file_format

        self.filename = "{}-{}.{}".format(title, length, file_format).lower().replace(' ', '-')


def get_campaigns(url):
    """Fetch campaign information and return list of Campaign objects"""
    # get HTML of campaign list and parse
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    # find all campaign links
    campaign_links = []
    for tag in soup.find_all("div", class_="CampaignPromo-media"):
        link = tag.a["href"]

        # prevent duplicates from being added
        if link not in campaign_links:
            campaign_links.append(tag.a["href"])

    campaigns = []
    for link in campaign_links:
        r = requests.get(link)
        soup = BeautifulSoup(r.text, "html.parser")

        # find all campaign IDs and names
        for tag in soup.find_all("div", class_="GroupPromo"):
            # only radio campaigns
            if tag["data-type"] == "Television":
                campaign = Campaign(tag["data-campaign-asset-group-id"], tag["data-campaign-name"])
                campaigns.append(campaign)

    return campaigns


def get_assets(campaigns, n, download_location):
    """Fetch asset information and return list of Asset objects"""
    assets = []
    count = 0
    curr_title = ''

    for campaign in campaigns:
        # Get JSON object
        r = requests.get(API_URL + campaign.id)

        for a in r.json():
            if count == n:
                return assets

            if a["language"] != "English":
                continue

            if "Market Specific" in a["title"] and (not a["marketArea"] or "MO" not in a["marketArea"]):
                continue

            # create asset object, stripping leading colon from length and making format lowercase
            asset = Asset(campaign, a["sourceUrl"], a["title"], a["length"][1:], a["fileFormat"].lower())

            file_path = os.path.join(download_location, asset.filename)
            if os.path.isfile(file_path):
                continue

            assets.append(asset)

            if asset.title != curr_title:
                count += 1

            curr_title = asset.title

    return assets


def download_assets(assets, download_location):
    """Download assets to specified location"""
    for asset in assets:
        path = download_location

        # get asset video
        r = requests.get(asset.url)

        # write video to file
        with open(os.path.join(path, asset.filename), "wb") as f:
            print("downloading {}".format(asset.filename))
            f.write(r.content)


def main():
    # Construct argument parse for command line interface
    parser = argparse.ArgumentParser(
        'Download videos from Ad Council campaigns for testing.')
    parser.add_argument('destdir', metavar='DESTDIR', type=str,
                        help='path to directory to download videos')
    parser.add_argument('-n', type=int, help='number of videos to download',
                        default=10)

    args = parser.parse_args()

    campaigns = get_campaigns(BROWSE_URL)
    assets = get_assets(campaigns, args.n, args.destdir)
    download_assets(assets, args.destdir)


if __name__ == "__main__":
    main()
