from playwright.sync_api import sync_playwright, BrowserContext

def get_datalayer(ctx: BrowserContext, url: str):
    page = ctx.new_page()
    page.goto(url)
    page.wait_for_load_state("networkidle")
    return page.evaluate("window.dataLayer")


with sync_playwright() as p:
    browser = p.chromium.launch()
    with browser.new_context() as bcon:
        data_layer = get_datalayer(bcon, "https://www.tayara.tn/")
        print(data_layer)