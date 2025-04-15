#(AIAutEnv) datascientist@Manjunaths-MacBook-Air AutomationPy % playwright codegen "https://www.tod.tv/en/home"

import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.tod.tv/en/home")
    page.get_by_role("button", name="Sign In").click()
    page.get_by_role("textbox", name="Email, username or phone").click()
    page.get_by_role("textbox", name="Email, username or phone").fill("datas@mail.com")
    page.get_by_role("textbox", name="Enter your password").click()
    page.get_by_role("textbox", name="Enter your password").fill("******")
    page.get_by_role("button", name="Sign in").click()
    page.locator(".relative > div > .relative").first.click()
    page.locator("#pin-input-0").fill("*")
    page.locator("#pin-input-1").fill("*")
    page.locator("#pin-input-2").fill("*")
    page.locator("#pin-input-3").fill("*")
    page.get_by_role("button", name=" Play").click()
    page.get_by_role("button", name="Close").click()
    page.goto("https://www.tod.tv/en/home")
    page.get_by_role("link", name="Movies").click()
    page.locator(".rail-item > .relative").first.click()
    page.get_by_role("link", name="TOD Picks").click()
    page.get_by_role("link", name="Lee").click()
    page.locator("#trailers").get_by_role("link").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
