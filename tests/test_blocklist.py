from retrieval.blocklist import domain_of, filter_results, filter_urls, is_blocked


def test_domain_strips_www_and_port():
    assert domain_of("https://www.Example.org:8080/a/b") == "example.org"


def test_known_fact_checkers_blocked():
    for url in [
        "https://www.politifact.com/factchecks/2024/x/",
        "http://snopes.com/fact-check/y",
        "https://fullfact.org/health/z",
    ]:
        assert is_blocked(url)


def test_subdomains_blocked():
    assert is_blocked("https://amp.politifact.com/a")


def test_substring_rule_catches_unlisted_factcheck_hosts():
    assert is_blocked("https://factcheck.example.com/a")
    assert is_blocked("https://fact-check.somenews.co.uk/a")


def test_ordinary_sources_pass():
    for url in [
        "https://en.wikipedia.org/wiki/X",
        "https://nature.com/articles/1",
        "https://bbc.co.uk/news/1",
    ]:
        assert not is_blocked(url)


def test_malformed_url_is_blocked():
    assert is_blocked("")
    assert is_blocked("not a url")


def test_filters():
    urls = ["https://snopes.com/a", "https://wikipedia.org/b"]
    assert filter_urls(urls) == ["https://wikipedia.org/b"]
    results = [{"url": u, "text": "t"} for u in urls]
    assert filter_results(results) == [{"url": "https://wikipedia.org/b", "text": "t"}]
