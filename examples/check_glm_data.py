""" This script accepts a list of filenames from the command line and attempts
to open each of them using `glmtools`. Because `glmtools` automatically performs
some flash-group-event parent-child calculations upon opening each file it is
a simple way to test for valid files. For instance, it confirms that each 
`event_parent_group_id` has a corresponding `group_id` entry.
"""

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import sys
from glmtools.io.glm import GLMDataset

filenames = sys.argv[1:]
for filename in filenames:
    try:
        glm = GLMDataset(filename)
    except KeyError as e:
        print(filename)
        logger.exception(e)