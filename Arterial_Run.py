import Rick_Arterial_Code as acode

scan1 = acode.collectData()
scan1 = acode.dataClean(scan1)
scan1 = acode.collectTimes(scan1)
scan1 = acode.determineDuration(scan1)
scan1 = acode.calculateBackground(scan1)
scan1 = acode.calculateAvgActLeft(scan1)
scan1 = acode.calculateMuCi(scan1)
acode.outputData(scan1)