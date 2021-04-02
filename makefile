test :
	docker build -qt wpbindt/allocator . && docker run -it --rm wpbindt/allocator python3 -m pytest -q
