from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
from six.moves import urllib
import numpy as np
import tensorflow as tf
from . import classify_image as cf

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
# Create your views here.
def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        human_string,score = cf.run_inference_on_image("/home/anirudh/classifyImages" + uploaded_file_url)
        return render(request,'upload/success.html',{"human_string":human_string,"score":score})
    return render(request,"upload/index.html",{})