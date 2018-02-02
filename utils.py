#!/usr/bin/env python

def _input(source, numlines = -1):
	# reads data from an input source and returns the decoded data
	encoding = 'big5hkscs'
	lines = []
	num_errors = 0
	counter = 0
	for line in open(source):
		if numlines > 0 and counter == numlines:
			break
		try:
			lines.append(line.decode(encoding))
		except UnicodeDecodeError as e:
			num_errors += 1
		counter += 1
	lines = ' '.join([line.rstrip('\n') for line in lines])

	print 'Finished decoding {} lines, encountered {} decoding errors and have a total of {} characters.'\
			.format(counter, num_errors, len(lines))
	return lines