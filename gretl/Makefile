PKG =  naiveFC

$(PKG).gfn: $(PKG).inp $(PKG).spec $(PKG)_sample.inp
	gretlcli --makepkg $(PKG).inp

#$(PKG).pdf: $(PKG).tex
#	pdflatex $<
#	bibtex $(PKG)
#	pdflatex $<
#	pdflatex $<

$(PKG).zip: $(PKG).gfn $(PKG).pdf
	echo makepkg $(PKG).zip | gretlcli -b -

install: $(PKG).zip
	echo pkg install $(PKG).zip --local | gretlcli -b -

clean:
	rm -f $(PKG).zip
