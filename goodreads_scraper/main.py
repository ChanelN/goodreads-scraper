from bs4 import BeautifulSoup

from selenium import webdriver

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
import pandas as pd
import pickle
import re


def delete_signin_prompt(driver):
    try:
        # get rid of prompt - only shows up sometimes
        signin_prompt = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.Overlay__close")))
        # text = signin_prompt.get_attribute("outerHTML")
        # print(text)

        close_button = signin_prompt.find_element(By.CSS_SELECTOR,
                                                  "div.Overlay__close button.Button.Button--transparent.Button"
                                                  "--small.Button--rounded")
        close_button.click()
        print("closed a popup window")
    except TimeoutException:
        print("no prompt")


def click_on_book(item, driver, book_details):
    page_link = item.find("a",
                          class_="a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal")
    page_link = page_link.get("href")
    driver.get("https://www.amazon.co.uk/" + page_link)
    time.sleep(1)
    direct_page_price(driver, None, book_details)


def find_book_format(results, book, find_format, book_details, driver):
    """

    :rtype: book_details which is the df of book_
    """
    for item in results:
        print("THIS IS BOOK ", book, "ON THE PAGE")

        title_div = item.find('a',
                              class_="a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal")

        current_title = ""
        # most of the time, the title is in one element, other times the title is split into multiple children classes
        children = title_div.find_all()
        for title in children:
            current_title += title.text

        time.sleep(2)

        is_sponsored = 'Sponsored' in item.find('span', {'class': 'a-color-base'}).text
        original_title = book_details['Title'].lower()
        current_title = current_title.lower()
        print("o", original_title, "new", current_title)
        if is_sponsored or original_title not in current_title:
            # continue skips the current item
            print("skipped")
            book += 1
            continue

        # BOOKS THAT ARENT SPONSORED
        # trying to find the individual book first - then formats for each book
        each_format = item.find_all("div", {
            "class": ["a-row a-spacing-mini", "a-row a-spacing-mini a-size-base a-color-base",
                      "a-section a-spacing-none a-spacing-top-small s-price-instructions-style"]})

        print(len(each_format))
        num = 0
        for format_type in each_format:
            # time.sleep(2)
            text = format_type.find("a",
                                    class_="a-size-base a-link-normal s-underline-text s-underline-link-text s-link-style a-text-bold")
            print(text.text, num)
            # text.text ==
            # "mass market paperback
            if find_format in text.text:
                print("found the format")
                # this will select the exact class instead of 'a-price a-text-price'

                # try the first section
                if num == 0:
                    print("first section")
                    first_format = item.find("div",
                                             class_="a-section a-spacing-none a-spacing-top-micro s-price-instructions-style")
                    if first_format is None:
                        first_format = item.find("div",
                                                 class_="a-section a-spacing-none a-spacing-top-small s-price-instructions-style")

                    book_price = first_format.find("span", class_="a-price")
                    if book_price is None:
                        # IF THERE'S PAPERBACK FORMAT, BUT NO PRICE ON RESULT PAGE, IT WILL CLICK ON LINK
                        click_on_book(item, driver, book_details)
                        break
                    else:
                        price_whole = book_price.find("span", class_="a-price-whole").text
                        price_decimal = book_price.find("span", class_="a-price-fraction").text
                        book_details["Format"] = find_format
                        book_details['Price'] = float(price_whole + price_decimal)
                    print(book_details['Price'])
                    break
                elif num > 0:
                    print("second")

                    book_price = format_type.find("span", class_="a-price")
                    if book_price is None:
                        click_on_book(item, driver, book_details)
                        break

                    price_whole = book_price.find("span", class_="a-price-whole").text
                    price_decimal = book_price.find("span", class_="a-price-fraction").text
                    book_details["Format"] = find_format
                    book_details['Price'] = float(price_whole + price_decimal)
                    print(book_details['Price'])
                    break
                # if there is a paperback label, but no price showing, skip
                else:
                    continue

            num += 1
        # OUTSIDE FOR LOOP FOR FORMATS
        book += 1

        if 'Price' in book_details:
            print('price in df')
            # it will break loop for other books
            break

    # outside of checking all books
    if 'Price' not in book_details:
        # check for hardcover format instead
        if find_format == "Paperback":
            print("checking for hardback now")
            book_details = find_book_format(results, 1, "Hardcover", book_details, driver)
        else:
            # if both paper and hardback not found
            book_details["Format"] = None
            book_details['Price'] = 0.0
    return book_details


def search_result_page_price(driver, htmlsoup, book_details):
    results = htmlsoup.find_all("div", {"data-component-type": "s-search-result"})
    book = 1

    book_details = find_book_format(results, book, "Paperback", book_details, driver)
    return book_details


def direct_page_price(driver, soup, book_details):
    price_classes = ["a-size-base a-color-secondary", "a-size-base a-color-price a-color-price", "a-color-base"]

    if soup is None:
        amazon_html = driver.page_source
        soup = BeautifulSoup(amazon_html, 'html.parser')

    all_formats = soup.find_all("a", class_="a-button-text")
    # goes through all available formats like kindle/audiobook, hardcover, paperback
    # this code needs to be used for case  1 too-UPDATE
    for book_format in all_formats:
        format_text = book_format.find("span")

        # animal farm has words 'Mass market paperback"
        if format_text and "Paperback" in format_text.text:
            print("found paperback")

            for class_name in price_classes:
                try:
                    # have to use class name, because the dom isn't the same for all pages either, anne of gg
                    price = book_format.find("span", class_=class_name).text
                    # anne of gg -> price = "from £9.34", only on direct page does this happen
                    pattern = r'(?:\d{1,3})(?:\.\d{1,2})'
                    match = re.search(pattern, price)
                    print("hey", match)
                    price = match.group(0)
                    book_details['Format'] = "Paperback"
                    book_details['Price'] = float(price)
                except AttributeError:
                    # continue onto next potential class name
                    continue
            break
    # if no paperback on direct page
    if 'Price' not in book_details:
        for book_format in all_formats:
            format_text = book_format.find("span")

            if format_text and "Hardcover" in format_text.text:
                print("Hardcover")
                book_details["Format"] = "Hardcover"
                try:
                    price = book_format.find("span", class_="a-size-base a-color-secondary").text
                except AttributeError:
                    try:
                        price = book_format.find("span", class_="a-size-base a-color-price a-color-price").text
                    except AttributeError:
                        price = book_format.find("span", class_="a-color-base").text

                # price = price.replace("£", "").strip()
                pattern = r'(?:\d{1,3})(?:\.\d{1,2})'
                match = re.search(pattern, price)
                price = match.group(0)
                book_details["Format"] = "Hardcover"
                book_details['Price'] = float(price)
                break
    if 'Price' not in book_details:
        amazon_search_for_book(driver, book_details)
    return book_details


def amazon_search_for_book(driver, book_price):
    searchbar = driver.find_element(By.CSS_SELECTOR, "input[placeholder='Search Amazon.co.uk']")
    current_search_text = searchbar.get_attribute("value")
    if current_search_text:  # if theres text
        searchbar.clear()

    searchbar.send_keys(book_price['Title'])
    search_button = driver.find_element(By.CSS_SELECTOR, "input[id='nav-search-submit-button']")
    search_button.click()
    time.sleep(5)

    # need to make a new soup from current page
    amazon_html = driver.page_source
    searched_soup = BeautifulSoup(amazon_html, 'html.parser')

    search_result_page_price(driver, searched_soup, book_price)
    # find_book_format(1, "Paperback", book_price, driver)


def get_book_price(driver, book_count):
    # SIGNIN PROMPT
    # delete_signin_prompt(driver)

    # find amazon button
    driver.find_element(By.CSS_SELECTOR, "button.Button--buy.Button--medium.Button--block").click()
    time.sleep(2)

    # delete cookies for amazon domain - since domain changes cookies from goodreads to amazon will make invalid errors
    driver.delete_all_cookies()

    # driver.switch_to.window(driver.window_handles[-1])
    # i can leave my og goodreads page open, and pass it to reviews scraper

    driver.close()
    # closes goodreads page and switches the driver to the amazon page
    driver.switch_to.window(driver.window_handles[0])

    # we're now on the amazon tab
    # here, some books return the amazon search page while others direct to a specific book href
    amazon_html = driver.page_source
    soup = BeautifulSoup(amazon_html, 'html.parser')
    book_price = {}
    book_price['Title'] = df.at[book_count, 'Title']

    # WHAT ABOUT WHEN THERE IS NO PAPERBACK EDITION
    # CASE 1: SEARCH PAGE SHOWS
    if soup.find("span", {"data-component-type": "s-search-results"}):
        print("returns search results page successfully")
        search_result_page_price(driver, soup, book_price)
        # find_book_format(1, "Paperback", book_price, driver)
    # CASE 2: SPECIFIC BOOK SHOWS
    elif soup.find("h1", {"id": "title"}):
        print("returns direct page")
        # text for formats is inside a span, that is child to an A element
        # paperback = soup.find("span", text="Paperback")
        direct_page_price(driver, soup, book_price)
    # CASE 3: LINK DOESN'T RETURN ANYTHING - need to manually search for book
    else:
        print("needs to be searched for")
        homepage = driver.find_element(By.CSS_SELECTOR, "b > a[href='/ref=cs_404_link']")
        link = homepage.get_attribute("href")
        # clicks on link to get to homepage, so we can search
        driver.get(link)
        amazon_search_for_book(driver, book_price)

    # close the book detail and amazon driver - outside if, elif
    # print("price outside", book_price['Price'])

    # why dont i combine the price to end of book info dataframe
    index = len(df_price)  # index for rows
    for key in book_price:
        df_price.at[index, key] = book_price[key]

    # print("df after adding price", df_price)
    # this is the final step - closes driver for the whole book
    driver.close()
    # return df_price
    return


def get_each_new_review_page(driver, data):
    # page_reviews = pd.DataFrame()

    # the divs use lazyload-wrapper, so we scroll to load reviews on the page
    # execute_script, pass in a Javascript command
    print("getting new page review")

    wait = WebDriverWait(driver, 30)
    # wait for all the article elements with the ReviewCard class to be present on the page.
    reviews = wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.ReviewCard")))
    time.sleep(2)

    for review in reviews:
        driver.execute_script("arguments[0].scrollIntoView(true);", review)
        review_data = {}

        try:
            #
            hidden_review = review.find_element(By.CSS_SELECTOR, "div.Alert.Alert--informational")
            if hidden_review is not None:
                # spoiler = True
                time.sleep(5)
                button = hidden_review.find_element(By.CSS_SELECTOR, "button.Button.Button--inline.Button--small")
                button.click()
        except NoSuchElementException or ElementClickInterceptedException:
            pass

        review_data['Title'] = data['Title']
        review_data['Username'] = review.find_element(By.CLASS_NAME, "ReviewerProfile__name").text
        # reviewCard row is inside of section "ReviewCard__content"
        review_data_div = review.find_element(By.CLASS_NAME, "ReviewCard__row")
        review_data['Review Date'] = review_data_div.find_element(By.CSS_SELECTOR, "span.Text.Text__body3").text

        # <span aria-label="Rating 5 out of 5" role="img" class="RatingStars RatingStars__small">
        shelf_div = review_data_div.find_element(By.CSS_SELECTOR, "div.ShelfStatus")
        try:  # reviews that don't have ratings
            all_stars = shelf_div.find_element(By.CSS_SELECTOR, "span.RatingStars.RatingStars__small")
            aria_label = all_stars.get_attribute("aria-label")
            rating_parts = aria_label.split(" ")
            review_data['Rating'] = int(rating_parts[1])
        except NoSuchElementException:
            review_data['Rating'] = None

        review_text = review.find_element(By.CSS_SELECTOR, "section.ReviewText")
        # filter out \n and symbols, html, punctuation, emoji
        review_data['Description'] = review_text.find_element(By.CSS_SELECTOR, "span.Formatted").text

        # it should be counting the len of page_reviews from outside
        # not locally passed from get_mulitple pages
        # taking in df_reviews means index is located at end of first book
        index = len(df_reviews)
        for key in review_data:
            df_reviews.at[index, key] = review_data[key]

    # print("df after adding 1 review page", df_reviews)
    return df_reviews


def get_n_new_page_reviews(driver, data):
    # clicks to go to next page, calls funct to get all reviews of that page out

    # NEED FUNCTION TO GET RID OF ALERT
    # delete_signin_prompt(driver)

    filter_div = driver.find_element(By.CSS_SELECTOR, "div.ReviewFilters")
    driver.execute_script("arguments[0].scrollIntoView();", filter_div)
    time.sleep(3)
    filter_div.find_element(By.CSS_SELECTOR, "button.Button.Button--secondary.Button--medium").click()

    time.sleep(5)
    overlay = driver.find_element(By.CLASS_NAME, "Overlay--floating")
    try:
        overlay.find_element(By.CSS_SELECTOR, "div.RadioGroup__input label[for='en']").click()
        overlay.find_element(By.CSS_SELECTOR,
                             "div.Overlay__actions button.Button.Button--primary.Button--small.Button--block").click()

    except NoSuchElementException:
        overlay.find_element(By.CSS_SELECTOR,
                             "div.Overlay__close button.Button.Button--transparent.Button"
                             "--small.Button--rounded").click()
    time.sleep(2)

    # the function get_each_new is what adds the reviews to the final pd
    # 1st page
    get_each_new_review_page(driver, data)

    # click more reviews button
    try:
        div = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.Divider.Divider--contents.Divider--largeMargin")))

        next_page = div.find_element(By.CSS_SELECTOR, "a[aria-label='Tap to show more reviews and ratings']")
        href = next_page.get_attribute("href")

        driver.get(href)
        time.sleep(3)
        # calling it again for the 2nd page of reviews
        get_each_new_review_page(driver, data)
    except TimeoutException:
        pass


def add_cookies(driver):
    # cookies for GOODREADS LOGIN TO GET UK OPTIONS
    with open("cookies.pkl", "rb") as file:
        cookies = pickle.load(file)
    for cookie in cookies:
        # print("cookie domain " + cookie["domain"])
        driver.add_cookie(cookie)

    time.sleep(1)
    driver.refresh()


def get_book_details(page_url):
    driver = webdriver.Chrome()
    driver.get(page_url)
    #
    #
    add_cookies(driver)
    #
    # occasionally, a prompt may popup
    # delete_signin_prompt(driver)

    data = {}

    amazon_html = driver.page_source
    time.sleep(2)
    book_soup = BeautifulSoup(amazon_html, 'html.parser')

    data['Title'] = book_soup.find("h1", {"data-testid": "bookTitle"}).text
    data['Author'] = book_soup.find("span", class_="ContributorLink__name").text

    meta_data = book_soup.find("div", class_="RatingStatistics__meta")
    rating_review_split = meta_data["aria-label"].split(" ")
    total_ratings = float(rating_review_split[0].replace(",", ""))
    total_reviews = float(rating_review_split[3].replace(",", ""))
    data['Total ratings'] = total_ratings
    data['Average rating'] = book_soup.find("div", class_="RatingStatistics__rating").text
    data['Total reviews'] = total_reviews

    description = book_soup.find("div", class_="DetailsLayoutRightParagraph")
    description_text = description.find("span", "Formatted").text
    data['Description'] = description_text

    all_genres = book_soup.find("div", class_="BookPageMetadataSection__genres")
    try:
        data['Top Genre'] = all_genres.find_all("span", class_="Button__labelItem")[0].text
    except AttributeError:
        data['Top Genre'] = 'Not listed'

    details = book_soup.find("div", class_="BookDetails")
    published_str = details.find("p", {"data-testid": "publicationInfo"}).text
    published = published_str[16:]
    data["Date published"] = published

    book_length = book_soup.find("p", {"data-testid": "pagesFormat"}).text
    book_length = book_length.split(" ")
    data["Pages"] = book_length[0]

    # swap back to driver
    social = driver.find_element(By.CSS_SELECTOR, "div.SocialSignalsSection")
    driver.execute_script("arguments[0].scrollIntoView(true);", social)
    # waits for content to load
    time.sleep(3)

    currently_reading = social.find_element(By.CSS_SELECTOR, "div[data-testid='currentlyReadingSignal']").text
    currently_reading = currently_reading.split(" ")[0]
    want_to_read = social.find_element(By.CSS_SELECTOR, "div[data-testid='toReadSignal']").text
    want_to_read = want_to_read.split(" ")[0]
    data["Currently reading"] = currently_reading
    data["Want to read"] = want_to_read

    index = len(df)
    current_book = index
    for key in data:
        # key = title, author, average rating, description, genre
        df.at[index, key] = data[key]

    # have to manually change the csv file name when scraping worst book listT
    df.to_csv("Test_data.csv", index=False)

    get_n_new_page_reviews(driver, data)
    df_reviews.to_csv("Test_reviews.csv", index=False)

    # I should get the book price after because i delete cookies for amazon
    get_book_price(driver, current_book)
    df_price.to_csv("Test_prices.csv", index=False)


# GOODREADS LOGIN VIA AMAZON INSTEAD-can trigger bot, periodically refresh>>?
def amazon_login():
    # the browser driver is in usr/local/bin
    login_driver = webdriver.Chrome()
    url = "https://www.goodreads.com"
    login_driver.get(url)

    nav_dropdown = login_driver.find_element(By.CSS_SELECTOR, "a.gr-button--amazon")
    href = nav_dropdown.get_attribute("href")
    login_driver.get(href)
    time.sleep(2)
    login_email = login_driver.find_element(By.ID, "ap_email")
    login_email.send_keys("chanels.testing@gmail.com")
    login_pass = login_driver.find_element(By.ID, "ap_password")
    login_pass.send_keys("Booktesting")
    login_driver.find_element(By.ID, "signInSubmit").click()
    time.sleep(2)
    # this is where the anti-bot gets triggered
    try:
        if login_driver.find_element(By.ID, 'auth-warning-message-box'):
            # have to manually bypass
            time.sleep(100)
    except:
        pass

    login_cookies = login_driver.get_cookies()
    with open("cookies.pkl", "wb") as file:  # this auto-closes file
        pickle.dump(login_cookies, file)

    login_driver.close()



# connecting and getting list of books
resp = requests.get("https://www.goodreads.com/list/show/1.Best_Books_Ever")

# control list
# resp = requests.get("https://www.goodreads.com/list/show/23974.Worst_Rated_Books_on_Goodreads")
# resp = requests.get("https://www.goodreads.com/list/show/2.The_Worst_Books_of_All_Time")

soup = BeautifulSoup(resp.text, "html.parser")
book_links = soup.find_all("a", class_="bookTitle")

# extract each books url from each <a>
# full url is made up of base url + /book/show/number.bookName(relative url)
base_link = "https://www.goodreads.com"
relative_url = [link['href'] for link in book_links]
df = pd.DataFrame()
df_reviews = pd.DataFrame()
df_price = pd.DataFrame()


# start of main program
amazon_login()
for link_no in range(1):
    page_url = base_link + relative_url[link_no]
    if link_no % 20 == 0:
        time.sleep(5)
    print(page_url, link_no)
    get_book_details(page_url)
print("all books finished")
