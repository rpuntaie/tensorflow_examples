
.PHONY: install
install:
	pip install --user --disable-pip-version-check -r requirements.txt

.PHONY: all
.DEFAULT_GOAL := all
all:
	advanced_autodiff.py
	audio.py
	audiodata.py
	autoencoder.py
	autograph.py
	clothes2clothes.py
	clothes2clothes_2.py
	clothes2clothes_3.py
	customize_fit.py
	data.py
	eager.py
	embed.py
	estimator.py
	gan1.py
	gan2.py
	mask.py
	prune.py
	quardratic.py
	ragged.py
	cd course_v1 && $(MAKE)
	cd course_v2 && $(MAKE)
	cd course_lite && $(MAKE)
