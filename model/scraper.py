from icrawler.builtin import GoogleImageCrawler

players = ["Taylor Swift"]

for player in players:
    crawler = GoogleImageCrawler(storage={'root_dir': f'images/{player.replace(" ", "_")}'})
    crawler.crawl(keyword=player, max_num=100)