
from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={"root_dir": "C:/ML_CEP/data/snacks"})
google_crawler.crawl(keyword="potato chips packets", max_num=1000)



google_crawler = GoogleImageCrawler(storage={"root_dir": "C:/ML_CEP/data/beverages"})
google_crawler.crawl(keyword="cold drink 2.5 litre", max_num=1000)


# google_crawler = GoogleImageCrawler(storage={"root_dir": "C:/FYP/phones"})
