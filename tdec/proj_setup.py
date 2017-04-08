__author__ = 'saguinag'+'@'+'nd.edu'
__version__ = "0.1.0"

##
## fname
##

## TODO: some todo list

## VersionLog:

import os, sys, time
import traceback

def main():
    # args = str(sys.argv)
    # print len(sys.argv)
    if len(sys.argv)<2:
        print "... Correct usage: python proj_setup.py [PATH TO HRG FILES]"
        os._exit(1)
    hrg_path = sys.argv[1]
    print hrg_path
    print "create sym links"

if __name__ == '__main__':
	try:
		main()

	except Exception, e:
		print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
		print str(e)
		traceback.print_exc()
		os._exit(1)
	sys.exit(0)
