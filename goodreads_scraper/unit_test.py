import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from main import amazon_login, add_cookies, base_link, relative_url, get_each_new_review_page, search_result_page_price
from bs4 import BeautifulSoup
import re


# before running test case, don't call any implementation code in main.py
class test_case(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()

    def test_a(self):
        amazon_login()
        assert True

    def test_b(self):
        self.driver.get("https://www.goodreads.com")
        add_cookies(self.driver)
        # pass assertion of element found
        self.assertTrue(self.driver.find_element(By.CSS_SELECTOR, "ul.personalNav").get_attribute("innerHTML"),
                        "cookies not added")

    def test_c(self):
        self.driver.get(base_link + relative_url[0])
        add_cookies(self.driver)
        book_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        self.assertTrue(book_soup.find("h1", {"data-testid": "bookTitle"}).text, "title not found")

    def test_d(self):
        pattern = r'(?:\d{1,3})(?:\.\d{1,2})'
        # text = "This books is £9.34"
        text = "This book is £1.34"
        match = re.search(pattern, text)
        print(match.group)
        price = match.group(1)
        expected = "1.34"
        self.assertEqual(price, expected, "This doesn't return the expected value")

    def test_e(self):
        self.driver.get(base_link + relative_url[0])
        add_cookies(self.driver)
        self.driver.find_element(By.CSS_SELECTOR, "button.Button--buy.Button--medium.Button--block").click()
        self.driver.delete_all_cookies()
        self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[0])
        book_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        self.assertTrue(book_soup.find("span", {"data-component-type": "s-search-results"}))

    def tearDown(self):
        # runs after case finished
        self.driver.close()


if __name__ == '__main__':
    unittest.main()
