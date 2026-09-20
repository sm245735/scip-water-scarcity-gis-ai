"""Observe official page rendering without modifying original download scripts."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/recovery/source_probe'

def main():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1600,'height':1000})
        errors=[]
        calls=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('response',lambda r:calls.append(dict(url=r.url,status=r.status)))
        try:
            page.goto('https://codis.cwa.gov.tw/StationData',wait_until='domcontentloaded',timeout=60000)
            page.wait_for_timeout(15000)
            (OUT/'codis_browser.html').write_text(page.content(),encoding='utf-8')
            (OUT/'codis_browser_text.txt').write_text(page.locator('body').inner_text(),encoding='utf-8')
            page.screenshot(path=str(OUT/'codis_browser.png'))
            print('Title:',page.title())
            print('Body:',page.locator('body').inner_text()[:1800])
        finally:
            (OUT/'codis_browser_network.json').write_text(json.dumps(dict(errors=errors,responses=calls),ensure_ascii=False,indent=2)+'\n')
            browser.close()

if __name__=='__main__':
    main()
