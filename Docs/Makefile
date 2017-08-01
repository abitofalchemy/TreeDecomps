CMD_SEP = "========================================"
CMD_DONE = "Done."
CLEAN_CMD = rm -f *.aux       ; \
            rm -f *.bbl       ; \
            rm -f *.blg       ; \
            rm -f *-blx.bib   ; \
            rm -f *.log       ; \
            rm -f *.out       ; \
            rm -f *.run.xml   ; \
            rm -f *.toc       ; \
            rm -f *.lof       ; \
            rm -f *.lot       ; \
            rm -f *.lol       ; \
            rm -f comment.cut ; \
            rm -f *.nav       ; \
            rm -f *.snm        ; \
            rm -f *.vrb        ; \

all: full

quick:
	@echo "QUICK BUILD"
	@echo
	@echo $(CMD_SEP)
	@echo "Building bibliography..."
	bibtex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo
	@echo $(CMD_SEP)
	@echo "Building poster_tdec..."
	lualatex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo

full:
	@echo "FULL BUILD"
	@echo
	@echo $(CMD_SEP)
	@echo "Building poster_tdec (1st time)..."
	lualatex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo
	@echo $(CMD_SEP)
	@echo "Building bibliography..."
	bibtex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo
	@echo $(CMD_SEP)
	@echo "Rebuilding poster_tdec (2nd time)..."
	lualatex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo
	@echo $(CMD_SEP)
	@echo "Rebuilding poster_tdec (3rd time)..."
	lualatex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo
	@echo $(CMD_SEP)
	@echo "Rebuilding poster_tdec (4th time)..."
	lualatex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo
	@echo $(CMD_SEP)
	@echo "Rebuilding poster_tdec (last time)..."
	lualatex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo

clean:
	@echo $(CMD_SEP)
	@echo "Cleaning '.' folder..."
	@-$(CLEAN_CMD)
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)

new: clean
	@echo "NEW BUILD"
	@echo
	@echo $(CMD_SEP)
	@echo "Building poster_tdec..."
	lualatex poster_tdec
	@echo $(CMD_DONE)
	@echo $(CMD_SEP)
	@echo
