.PHONY: help check compile test

help:
	@printf '%s\n' \
		'Kullanılabilir komutlar:' \
		'  make check    Repo yapısını ve Python sözdizimini kontrol eder' \
		'  make compile  Python dosyalarını derleme kontrolünden geçirir' \
		'  make test     Yerel yol ve manifest akışlarını sınar'

check:
	python3 scripts/check_repository.py

compile:
	python3 -m compileall -q -x '(^|[/\\])\._' aasist-source scripts

test:
	python3 -m unittest discover -s tests
