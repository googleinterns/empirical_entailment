import pandas as pd
import time
import json
import os

from typing import Any

from bs4 import BeautifulSoup
from selenium import webdriver


def browser_driver_setup(args):
    """
    Set up the browser
    :param args: Args for browser specific configs for selenium
    :return:
    """
    option = webdriver.ChromeOptions()
    option.add_argument(' — incognito')
    browser = webdriver.Chrome()
    return browser


def get_ad_info_from_link(url: str,
                          browser: webdriver.Chrome):
    """
    With the link to ads page on Google Transparency Report (from the csv), get the assets, vis_url and description
    for the ads
    :param url:
    :param browser:
    :return:
    """
    browser.get(url)
    time.sleep(1.5)
    el_ad_ctnr = browser.find_elements_by_class_name('ad-container')
    if not el_ad_ctnr:
        return None

    soup_ad_ctnr = BeautifulSoup(el_ad_ctnr[0].get_attribute("innerHTML"), 'html.parser')

    _all_divs = soup_ad_ctnr.find_all("div")

    if len(_all_divs) != 4:
        print("Malformed Webpage: {}".format(url))
        return None

    assets = _all_divs[0].get_text()
    url = _all_divs[1].get_text()
    url = url.split(" ")[1]
    desc = _all_divs[3].get_text()

    return assets, url, desc


def get_lp_content_from_vis_url(vis_url: str) -> Any:
    """
    Using the vis_url to identify the LP + parse useful text segments from the LP.
    TODO: Not included here, as the parser I implemented didn't work well
    :param vis_url:
    :return:
    """
    pass


def main(args) -> None:
    """
    Parse all js-rendered Google Transparency ads page and get the assets, vis_url and description

    :param args.creative_stats: Path to creative_stats.csv, downloaded from
    https://storage.googleapis.com/transparencyreport/google-political-ads-transparency-bundle.zip

    :param args.output_path: Path to an output json file with results
    :return: None
    """
    df = pd.read_csv(args.creative_stats)
    orig_len = len(df.index)
    print("Loaded stats for {} ads...".format(orig_len))

    browser = browser_driver_setup(args)
    total_count = 0
    valid_count = 0

    last_ad_id = None
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as fin:
            for line in fin:
                _ad_obj = json.loads(line)
                last_ad_id = _ad_obj['ad_id']

        last_df_idx = df.loc[df["Ad_ID"] == last_ad_id].index.values.astype(int)[0]

        df = df[df.index > last_df_idx]
        print("Loaded {} ads from existing output at {}...".format(orig_len - len(df.index), args.output_path))

    with open(args.output_path, 'a') as fout:
        for index, row in df.iterrows():
            total_count += 1
            _ad_id = row['Ad_ID']

            _url = row['Ad_URL']
            _ad_type = row['Ad_Type']
            _ad_regions = row['Regions']

            if _ad_type == 'Text' and _ad_regions == 'US':
                res = get_ad_info_from_link(url=_url, browser=browser)
                if res is None:
                    continue

                _assets, _vis_url, _desc = res[0], res[1], res[2]
            else:
                continue

            obj = {
                "assets": _assets,
                "vis_url": _vis_url,
                "desc": _desc,
                "ad_id": _ad_id,
                "post_date": row['Date_Range_Start'],
                "expire_date": row['Date_Range_End'],
            }
            obj_str = json.dumps(obj)
            fout.write(obj_str)
            fout.write("\n")

            valid_count += 1

            if valid_count % 100 == 0:
                print("Processed: {} valid text ads among {} total.".format(valid_count, total_count))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("creative_stats", type=str, help="path to google-political-ads-creative-stats.csv")
    parser.add_argument("output_path", type=str, help="path to output")
    args = parser.parse_args()

    main(args)