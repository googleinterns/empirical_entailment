### Political Ads (WIP)

Here contains a WIP script for crawling metadata + LP content for poltiical text ads listed in the [Google Transparency Report](
https://transparencyreport.google.com/political-ads/region/US). 

As the site is JavaScript rendered, we use the [Selenium](https://selenium-python.readthedocs.io/) WebDriver as an 
interface to browser on your computer/server (e.g. Chrome) to render the page before crawling.

#### Setup 

With a `python=3.7` virtual environment, install the packages in `requirements.txt`:
```
$ pip install -r requirements.txt
```

Follow instructions on [selenium doc](https://selenium-python.readthedocs.io/installation.html#drivers) to install the 
corresponding web driver for your browser. In case of Chrome or any Chromium-based browser, look at [this page](https://selenium-python.readthedocs.io/installation.html#introduction).

After setting up the browser driver, download the [Google Ads Transparency Bundle](https://storage.googleapis.com/transparencyreport/google-political-ads-transparency-bundle.zip). Unzip the file.

Start crawling the pages.

```
$ python download_ads_lp --creative_stats <path to creative_stats.json in the unziped bundle> --output_path <output_path>
```

Under default setup (e.g. sleep time), it takes about 12 hours to finish crawling all pages. 