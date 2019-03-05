# Core GUI libraries and userform information
import classipy2_ui
from PyQt4 import QtCore, QtGui, QtOpenGL
import sys

# Numeric processing tools
import numpy as np
from math import acos, sin, radians, degrees
import pandas as pd

# Geoprocessing tools
from osgeo import gdal
import pyproj

# Image processing
from PIL import Image, ImageQt
from skimage.feature import greycomatrix, greycoprops, canny
from skimage import data

# Machine learning 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.preprocessing import normalize

# Database
import mysql.connector

# Other
import pydot

#######Tools to be implemented#######

###Database Support

###Need to be able to create and edit databases
###View database visually
###Implement database control via command line

###Command Line

###Mostly for database control


class classiPy2_Form(QtGui.QMainWindow, classipy2_ui.Ui_MainWindow):

    def __init__(self, parent=None):

        super(classiPy2_Form, self).__init__(parent)

        self.setupUi(self)

        #######Set up the Main Graphics view#######
        self.drawArea = canvas()
        self.rasterView.addWidget(self.drawArea)
        self.drawArea.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)

        #######Set up the Preview Graphics view#######
        self.previewArea = preview()
        self.previewView.addWidget(self.previewArea)
        
        #######Set up the tree view#######
        self.rTree = rasterTree()
        self.treeView.addWidget(self.rTree)

        #######Set up raster property table#######
        self.rTable = rasterPropertyTable()
        self.tableView.addWidget(self.rTable)

        #######Set up message log / command line#######
        self.cmdLine = cmdLine()
        self.cmdView.addWidget(self.cmdLine)

        #######Set up active database view#######
        self.currentDatabase = currentDatabase()
        self.databaseView.addWidget(self.currentDatabase)
        
        #######Signals#######
        self.actionImport.triggered.connect(self.importRaster)
        self.actionExport.triggered.connect(self.exportRaster)

        self.connect(self.toolImport, QtCore.SIGNAL("clicked()"), self.importRaster)
        self.connect(self.toolExport, QtCore.SIGNAL("clicked()"), self.exportRaster)
        
        self.actionDeleteSelected.triggered.connect(self.deleteSelected)
        self.connect(self.toolDeleteRaster, QtCore.SIGNAL("clicked()"), self.deleteSelected)
        self.contrastSlider.sliderReleased.connect(self.changeContrast)
        
        self.connect(self.actionCommitGraphics, QtCore.SIGNAL("clicked()"), self.drawArea.commitPatches)
        self.connect(self.actionClearGraphics, QtCore.SIGNAL("clicked()"), self.drawArea.clearPatches)
        self.connect(self.actionRandomForest, QtCore.SIGNAL("clicked()"), self.rfClassify)

        self.connect(self.actionClassifyRaster, QtCore.SIGNAL("clicked()"), self.drawArea.moving_patch)

        # Database related signals
        self.connect(self.actionConnectDb, QtCore.SIGNAL("clicked()"), self.openClassipyDb)
        self.connect(self.actionDisconnect, QtCore.SIGNAL("clicked()"), self.closeClassipyDb)
        self.connect(self.actionSetActiveDb, QtCore.SIGNAL("clicked()"), self.setActiveDatabase)
        self.connect(self.actionCreateNewDb, QtCore.SIGNAL("clicked()"), self.createNewDatabase)
        self.connect(self.actionSetActiveTable, QtCore.SIGNAL("clicked()"), self.setActiveTable)
        self.connect(self.actionTest, QtCore.SIGNAL("clicked()"), self.sql_to_df)
        self.connect(self.toolImportBathy, QtCore.SIGNAL("clicked()"), self.loadBathy)
        self.connect(self.toolImportCurrents, QtCore.SIGNAL("clicked()"), self.loadCurrentModel)

        #######Coordinate Info for Patch Selection#######
        #######Might not need these, will create separate object for each patch#######
        self.veg_patches = []
        self.roc_patches = []
        self.sed_patches = []
        self.patches = []
        self.patch_size = self.patchSpinBox.value()

        #######Point storage for current model#######
        self.currents = pd.DataFrame(columns=['x','y','v'])

        #######Database setup#######
        self.glcm_dataframe = pd.DataFrame(columns=['type', 'subtype','correlation','dissimilarity','homogeneity','energy','contrast','ASM', 'entropy', 'rugosity'])
        self.db = None
        self.cursor = None
        self.activeDatabase = None
        self.activeTbl = None

        #######Bathymetry Data#######
        self.bathy = None
        self.bathyCoords = None
        self.bathyGridSize = None
        self.bathyPixmap = None

        #######Variables#######
        self.rasterPixmaps = []
        self.rasterItems = []
        self.rasterNames = []
        self.rasterArray = []
        self.treeItemNum = 0
        self.rasterGeo = []
        self.rasterCoords = []
        self.rasterBandNums = []
        self.activeRasterIndex = 0
        self.rfAccuracy = 0

        #######Just some introductory stuff in the log#######

        self.cmdLine.updateLog("ClassiPy V2.0.0")

    #######Texture Based Classification#######

    def getGreyLevCoMatrix(self, patch):

        return(greycomatrix(patch, [5], [0], 256, symmetric = True, normed = True))      

    def rfClassify(self):

        self.rfGenericSubstrateClassifier(self.glcm_dataframe)

    def rfGenericSubstrateClassifier(self, dataframe):

        #encodedDataframe = dataframe.join(pd.get_dummies(dataframe, columns=['type', 'subtype']))

        #To keep my own train of thought on the track, here is a description of what is / will be going on here:
        #The dataframe containing the bottom type, subtype, and GLCM properties are split into
        #Feature and label structures.
        #A series of arrays are made in order to test the accuracy of the training images which
        #are to be stored in the database. WE ARE NOT YET ACTUALLY CLASSIFYING ANYTHING, SIMPLY
        #TRAINING THE CLASSIFIER.
        #Make a prediction on the test data in order to retrieve an accuracy metric of the data. Again
        #this is just for testing the training data.
        #The selected data, if it meets a certain threshold of accuracy (I am thinking 80%), is saved to
        #a local database.
        #In another function, the Random Forest clasifier will be used once again, however instead of
        #utilizing the train_test_split function, the database data will be passed to the fit function
        #for training, and patches from the active raster will be passed in as test data. Note that rather
        #than the user selecting individual unknown patches, that classification will occur over the
        #entire raster in blocks.
        features = dataframe[['correlation','dissimilarity','homogeneity','energy','contrast','ASM','entropy', 'rugosity']]

        labels = dataframe['type']

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3)

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(train_features, train_labels)

        rfPredict = rf.predict(test_features)

        #Take the predicted types and append them to the dataframe
        test_features['type'] = rfPredict

        #We need to take the outcome and encode them for subtype classification
        encoded_types = test_features.join(pd.get_dummies(test_features['type']))

        self.rfAccuracy = metrics.accuracy_score(test_labels, rfPredict) * 100

        importances = list(rf.feature_importances_)

        self.cmdLine.updateLog("RF Classification Accuracy: " + str(self.rfAccuracy))

        #tree_sample = rf.estimators_[5]
        #export_graphviz(tree_sample, out_file = 'tree_sample.dot', feature_names = list(features.columns), rounded = True, precision =1)

        #(graph,) = pydot.graph_from_dot_file('tree_sample.dot')

        #graph.write_png('tree_sample.png')

        #print(importances)
        #print(self.rfAccuracy)
        #print(rfPredict, test_labels)

    def glcmPropsToDb(self):

        # Lists to store the GLCM properties
        correlation = []
        dissimilarity = []
        homogeneity = []
        energy = []
        contrast = []
        asm = []
        entropy = []
        rugosity = []
        Type = []
        subtype = []

        # Create GLCM (self.getGreyLevCoMatrix()), extract properties (greycoprops()), and append each property to its corresponding list.
        for p in self.patches:

            correlation.append(greycoprops(self.getGreyLevCoMatrix(p.get_patch()), 'correlation')[0,0])
            dissimilarity.append(greycoprops(self.getGreyLevCoMatrix(p.get_patch()), 'dissimilarity')[0,0])
            homogeneity.append(greycoprops(self.getGreyLevCoMatrix(p.get_patch()), 'homogeneity')[0,0])
            energy.append(greycoprops(self.getGreyLevCoMatrix(p.get_patch()), 'energy')[0,0])
            contrast.append(greycoprops(self.getGreyLevCoMatrix(p.get_patch()), 'contrast')[0,0])
            asm.append(greycoprops(self.getGreyLevCoMatrix(p.get_patch()), 'ASM')[0,0])
            Type.append(str(p.get_type()))
            subtype.append(str(p.get_subtype()))
            entropy.append(str(p.get_entropy()))
            rugosity.append(str(p.get_rugosity()))

        self.glcm_dataframe['type'] = pd.Series(Type)
        self.glcm_dataframe['subtype'] = pd.Series(subtype)
        self.glcm_dataframe['correlation'] = pd.Series(correlation)
        self.glcm_dataframe['dissimilarity'] = pd.Series(dissimilarity)
        self.glcm_dataframe['homogeneity'] = pd.Series(homogeneity)
        self.glcm_dataframe['energy'] = pd.Series(energy)
        self.glcm_dataframe['contrast'] = pd.Series(contrast)
        self.glcm_dataframe['ASM'] = pd.Series(asm)
        self.glcm_dataframe['entropy'] = pd.Series(entropy)
        self.glcm_dataframe['rugosity'] = pd.Series(rugosity)

        self.currentDatabase.add_data(self.glcm_dataframe)

    #######Database stuff##############
    def openClassipyDb(self):

        host = str(self.dbHostEdit.text())
        user = str(self.dbUserEdit.text())
        passwd = str(self.dbPasswdEdit.text())

        try:

            self.cmdLine.updateLog('Connecting to SQL Server...')
            self.db = mysql.connector.connect(host=host, user=user, passwd=passwd)
            self.cursor = self.db.cursor()
            self.cmdLine.updateLog('Connected.')
            self.actionDisconnect.setEnabled(True)

            p = self.dbStatus.palette()
            p.setColor(self.dbStatus.backgroundRole(), QtCore.Qt.green)
            self.dbStatus.setPalette(p)

            self.dbStatus.setText("Database Connected")

            self.comboDatabases.setEnabled(True)
            self.actionSetActiveDb.setEnabled(True)

            self.updateAvailableDatabases()

        except Exception as e:

            self.cmdLine.updateLog(str(e))

    def closeClassipyDb(self):

        self.cursor.close()
        self.db.close()

        p = self.dbStatus.palette()
        p.setColor(self.dbStatus.backgroundRole(), QtCore.Qt.red)
        self.dbStatus.setPalette(p)
        self.dbStatus.setText("Database Disconnected")
        self.cmdLine.updateLog("Disconnected from SQL Server.")
        
        self.actionDisconnect.setEnabled(False)

    def updateAvailableDatabases(self):

        #Databases implicit to sql. We do not want to display these.
        locked_db = ['information_schema','mysql','performance_schema','sakila','sys', 'world']
        
        show_dbs = "SHOW DATABASES"

        self.cursor.execute(show_dbs)

        self.comboDatabases.clear()

        for x in self.cursor:

            if x[0] not in locked_db:
                
                self.comboDatabases.addItem(x[0])

    def updateAvailableTables(self):

        show_tables = "SHOW TABLES"

        self.cursor.execute(show_tables)

        self.comboTables.clear()

        for x in self.cursor:

            self.comboTables.addItem(x[0])

    def setActiveDatabase(self):

        sql_setDb = "USE " + str(self.comboDatabases.currentText())

        self.cursor.execute(sql_setDb)

        self.activeDatabase = str(self.comboDatabases.currentText())

        self.activeDb.setText(self.activeDatabase)

        self.updateAvailableTables()

        self.cmdLine.updateLog("Active database set to " + str(self.comboDatabases.currentText()))

    def setActiveTable(self):

        self.activeTable.setText(str(self.comboTables.currentText()))

        self.activeTbl = str(self.activeTable.text())

        self.cmdLine.updateLog("Active table set to " + str(self.comboTables.currentText()))

    def createNewDatabase(self):

        pass

    def createNewTable(self):

        pass

    def classify_to_sql(self, dataframe = pd.DataFrame()):

        cursor = self.db.cursor(buffered=True)
        
        try:

            for i in range(len(dataframe['type'])):
                
                sql_insert = "INSERT INTO " + str(self.activeTbl) + " (type, subtype, correlation, dissimilarity, homogeneity, energy, contrast, asm, entropy, rugosity) VALUES ('%s','%s',%s,%s,%s,%s,%s,%s,%s,%s)" % (dataframe.at[i,'type'],dataframe.at[i,'subtype'], dataframe.at[i, 'correlation'], dataframe.at[i, 'dissimilarity'], dataframe.at[i, 'homogeneity'], dataframe.at[i, 'energy'], dataframe.at[i, 'contrast'], dataframe.at[i, 'ASM'], dataframe.at[i, 'entropy'], dataframe.at[i, 'rugosity'])
        
                cursor.execute(sql_insert)

                self.db.commit()

            #cursor.execute("SELECT id, type, correlation FROM classipy")

        except Exception as e:

            self.db.rollback()

            print(e)

        cursor.close()

    def sql_to_df(self):

        cols = ['type','subtype','correlation','dissimilarity','homogeneity','energy','contrast','asm', 'entropy', 'rugosity']

        df = pd.DataFrame(columns=['type', 'subtype', 'correlation','dissimilarity','homogeneity','energy','contrast','asm', 'entropy', 'rugosity'])

        for name in cols:

            sql_pull = "SELECT " + str(name) + " FROM " + str(self.activeTbl)

            self.cursor.execute(sql_pull)

            vals = []

            for x in self.cursor:

                #print(x[0])

                vals.append(x[0])

            df[str(name)] = pd.Series(vals)

            vals = []

        return(df)

    #######End database tools#######

    def changeContrast(self):

        print(str(self.contrastSlider.value()))

        i = None
        p = 0

        for x in self.rTree:

            if x.checkState(0) == QtCore.Qt.Checked:

                i = self.rTree.indexFromItem(x)
                p += 1

        if p > 1:

            return

        else:

            pass
        
    def deleteSelected(self):

        i = None

        for x in self.rTree:

            if x.checkState(0) == QtCore.Qt.Checked:

                i = self.rTree.indexFromItem(x)

                i = i.row()

                self.rTree.takeTopLevelItem(i)

        self.drawArea.scene().removeItem(self.rasterItems[i])
                
        self.rasterPixmaps.pop(i)
        self.rasterNames.pop(i)
        self.rasterItems.pop(i)
        self.treeItemNum -= 1
        self.rasterGeo.pop(i)
        self.rasterCoords.pop(i)
        self.rasterBandNums.pop(i)
        self.rasterArray.pop(i)
        self.activeRasterIndex -= 1
        self.drawArea.clearPatches()

    def updateQGraphic(self):

        pass

    def parseGeodetics(self, geodetics):

        filteredGeo = geodetics

        for p in ['[', ']', '"']:

            filteredGeo = filteredGeo.replace(p, '')

        filteredGeo = filteredGeo.split(',')

        return(filteredGeo)

    def loadCurrentModel(self):

        inputCurrentFile = str(QtGui.QFileDialog.getOpenFileName(self, 'Select current model file...', '', ".dat(*.dat)"))
        inputCurrentLocFile = str(QtGui.QFileDialog.getOpenFileName(self, 'Select current model location file...', '', ".dat(*.dat)"))

        crs_wgs = pyproj.Proj(init='epsg:4326')
        crs_utm = pyproj.Proj(init='epsg:26909')

        currents = pd.read_csv(inputCurrentFile, sep='\t', header=None)
        currents.columns = ['velocity','depth','notImportant']
        currents.drop(['depth', 'notImportant'], axis=1)

        locations = pd.read_csv(inputCurrentLocFile, sep=' ', header=None)
        locations.columns = ['lat', 'long']
        #locations['velocity'] = currents['velocity']

        lat = np.array(locations['lat'])
        lon = np.array(locations['long'])

        x, y = pyproj.transform(crs_wgs, crs_utm, lat, lon)

        self.currents['x'] = x
        self.currents['y'] = y
        self.currents['v'] = currents['velocity']

        self.cmdLine.updateLog("Ocean current model loaded successfully.")
  
    def loadBathy(self):

        inputBathyFilePath = QtGui.QFileDialog.getOpenFileName(self, 'Select file to open...', '', ".tif(*.tif)")
        rasterCoords = None
        noData = 3.4028230607371e+38
        
        
        rasterFile = gdal.Open(str(inputBathyFilePath))

        rasterCoords = (rasterFile.GetGeoTransform()[0], rasterFile.GetGeoTransform()[3])

        self.bathyGridSize = rasterFile.GetGeoTransform()[1]

        rasterBand = rasterFile.GetRasterBand(1)

        #arr = rasterBand.SetNoDataValue(noData)

        arr = np.array(rasterBand.ReadAsArray())

        arr = np.ma.masked_where(arr == abs(noData), arr)

        self.bathy = arr
        self.bathyCoords = rasterCoords

        self.cmdLine.updateLog("Bathymetry loaded into memory. \n" + "Corner at XY: " + str(rasterCoords[0]) + 'm, ' + str(rasterCoords[1]) + 'm')

        self.bathyPixmap = self.rasterToQPixMap(inputBathyFilePath)
        
        #val[0] will output the array
        #val[1][0] will output the x coordinate of the top left corner
        #val[1][1] will output the y coordinate of the top left corner

    def rasterToQPixMap(self, inputRasterFilePath):

        # extract the file name from the file path
        filename = inputRasterFilePath.split("/")[-1]

        # Add the raster name as a top level tree object in the raster tree view
        tlItem = QtGui.QTreeWidgetItem(self.rTree, [str(filename)])

        tlItem.setCheckState(0, QtCore.Qt.Unchecked)
        
        self.rTree.addTopLevelItem(tlItem)

        self.rTree.setItems()

        # Store the raster name into a global list
        self.rasterNames.append(str(inputRasterFilePath))

        # Load the raster using Gdal
        rasterFile = gdal.Open(str(inputRasterFilePath))

        #Extract the raster information and save to a list
        rasterGeo = rasterFile.GetProjection()[6:]

        cleanedRasterGeo = self.parseGeodetics(rasterGeo)

        self.rasterGeo.append(cleanedRasterGeo)

        self.rasterCoords.append((rasterFile.GetGeoTransform()[0], rasterFile.GetGeoTransform()[3]))
                                 
        # Retrieve the total number of bands in the raster. This will allow us to make an array of each individual raster
        # in the next step.
        bandNum = rasterFile.RasterCount
        
        self.rasterBandNums.append(bandNum)

        # A list in which the band names will be stored.
        pixmapBands = []

        # Save a different pixmap for each individual band.
        # The process for converting a Gdal raster into a pixmap is as such:
            # 1. Convert a selected raster band into a numpy array
            # 2. Convert the raster values to unsigned 8 bit integers as this is what is required for a pixmap
            # 3. Generate a PIL image from the uint8 raster array
            # 4. Convert the PIL image into a QImage
            # 5. Finally convert the QImage into a QPixmap which is then appended to a list where it can be accessed.
        for x in range(bandNum):

            rasterBand = rasterFile.GetRasterBand(x + 1)

            raster_pixmap = self.rasterToArray(rasterBand, 5)

            pixmapBands.append(raster_pixmap)

        # Add a child QTreeWidgetObject for each individual band in the raster.
        for bands in range(bandNum):

            QtGui.QTreeWidgetItem(self.rTree.topLevelItem(self.treeItemNum), [str('Band ' + str(bands + 1))])

        # Save it to a local list before appending it to the global list. This will allow for easier management
        # when multiple rasters are loaded in.
        self.rasterPixmaps.append(pixmapBands)

        self.treeItemNum += 1

    def rasterToArray(self, inputRaster, scaleFactor = 5):

        # Set nodata values to a more standard value
        noData = -3.4028230607371e+38

        arr = inputRaster.SetNoDataValue(noData)

        arr = np.array(inputRaster.ReadAsArray())

        arr = np.ma.masked_where(arr == noData, arr)

        arr = (arr * scaleFactor).astype(np.uint8)

        self.rasterArray.append(arr)

        arr_Image = Image.fromarray(arr)

        arr_Image = QtGui.QImage(ImageQt.ImageQt(arr_Image))

        arr_pixmap = QtGui.QPixmap.fromImage(arr_Image)

        return(arr_pixmap)

    def drawItem(self, pixmap):

        self.drawArea.empty = False

        self.drawArea.photo = QtGui.QGraphicsPixmapItem(pixmap)

        self.rasterItems.append(self.drawArea.photo)

        # Determines the coordinates at which the image will be placed.       
        self.drawArea.photo.setOffset(self.rasterCoords[self.activeRasterIndex][0], self.rasterCoords[self.activeRasterIndex][1])

        self.drawArea.scene().addItem(self.drawArea.photo)

        self.rTable.setRasterProperties((len(self.rasterNames) - 1))

        self.activeRasterIndex += 1

    # You may be wondering why I have two drawing functions. Why not pass graphicsArea as an arguement?
    # The other draw function is also responsible for updating the tree view and updates the raster index.
    # The other one also sets the image to proper coordinates. This is wholly unnecessary for the preview.
    # I'd rather copy/paste the function sans those lines for the previewView (10/10 name) rather than
    # implement some conditional statement to take care of it.
    def drawPreview(self, pixmap):

        self.previewArea.photo = QtGui.QGraphicsPixmapItem(pixmap)

        self.previewArea.fitInView(self.previewArea.photo)

        self.previewArea.scene().addItem(self.previewArea.photo)
        

    def importRaster(self):

        # Not enough support for multi raster editing currently, so lets just disable that.
        if self.activeRasterIndex == 1:

            return

        # Open a QFileDialog so that a tiff file can be loaded
        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Select file to open...', '', ".tif(*.tif)")

        # Inform the raster graphic view that it will no longer be empty.
        # this will enable the pan and zoom functions
        self.drawArea.empty = False

        # Feed the raster filepath into the rasterToQPixMap function
        self.rasterToQPixMap(fileName)

        # Update the raster graphics viewer with the newly loaded raster.
        self.drawItem(self.rasterPixmaps[self.activeRasterIndex][0])

    def exportRaster(self):

        return(QtGui.QFileDialog.getSaveFileName(self))

class currentDatabase(QtGui.QTableWidget):

    def __init__(self):

        super(currentDatabase, self).__init__()

        self.setColumnCount(8)

        hHeader = self.horizontalHeader()
        hHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.setHorizontalHeaderLabels(['Type','Subtype','Correlation','Dissimilarity','Homogeneity','Energy','Contrast','ASM', 'Entropy', 'Rugosity'])

    def add_data(self, dataframe):

        for x in range(dataframe.shape[0]):
            
            self.insertRow(x)

        for x in range(dataframe.shape[1]):

            for y in range(dataframe.shape[0]):

                try:

                    self.setItem(y, x, QtGui.QTableWidgetItem(str(round(dataframe.iloc[y, x], 2))))

                except:

                    self.setItem(y, x, QtGui.QTableWidgetItem(str(dataframe.iloc[y, x])))

class sqlDatabase(QtGui.QTableWidget):

    def __init__(self):

        super(currentDatabase, self).__init__()

        self.setColumnCount(7)

        hHeader = self.horizontalHeader()
        hHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.setHorizontalHeaderLabels(['Type','Subtype','Correlation','Dissimilarity','Homogeneity','Energy','Contrast','ASM', 'Entropy', 'Rugosity'])

class rasterPropertyTable(QtGui.QTableWidget):

    def __init__(self):

        super(rasterPropertyTable, self).__init__()

        self.setColumnCount(2)
        
        hHeader = self.horizontalHeader()
        hHeader.setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        hHeader.setResizeMode(1, QtGui.QHeaderView.ResizeToContents)

        vHeader = self.verticalHeader()
        vHeader.setResizeMode(QtGui.QHeaderView.ResizeToContents)

        self.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        
        properties = ['Name:', 'X:', 'Y:', 'Bands: ','Projection:', 'Central Meridian: ', 'Scale Factor: ', 'Spheroid: ']

        for x in range(len(properties)):

            self.insertRow(x)

            self.setItem(x, 0, QtGui.QTableWidgetItem(str(properties[x])))

    def get_parent(self):

        return(self.parent().parent().parent().parent().parent().parent().parent().parent())
        
    def setRasterProperties(self, rNum):

        self.setItem(0,1, QtGui.QTableWidgetItem(self.get_parent().rasterNames[rNum]))
        self.setItem(1,1, QtGui.QTableWidgetItem(str(self.get_parent().rasterCoords[rNum][0])))
        self.setItem(2,1, QtGui.QTableWidgetItem(str(self.get_parent().rasterCoords[rNum][1])))
        self.setItem(3,1, QtGui.QTableWidgetItem(str(self.get_parent().rasterBandNums[rNum])))
        self.setItem(4,1, QtGui.QTableWidgetItem(str(self.get_parent().rasterGeo[rNum][0])))
        self.setItem(5,1, QtGui.QTableWidgetItem(str(self.get_parent().rasterGeo[rNum][14])))
        self.setItem(6,1, QtGui.QTableWidgetItem(str(self.get_parent().rasterGeo[rNum][16])))
        self.setItem(7,1, QtGui.QTableWidgetItem(str(self.get_parent().rasterGeo[rNum][3][8:])))
            

class rasterTree(QtGui.QTreeWidget):

    def __init__(self):

        super(rasterTree, self).__init__()
        
        self.setAlternatingRowColors(True)

        self.setHeaderLabels(['Raster Files'])

        self.items = []

    def setItems(self):

        iter = QtGui.QTreeWidgetItemIterator(self, flags=QtGui.QTreeWidgetItemIterator.All)

        while iter.value():

            item = iter.value()
            
            self.items.append(item)

            iter += 1

    # Allow us to iterate through the QTreeWidget to more intuitively interact with the items.
    def __iter__(self):

        self.i = 0

        return(self)

    def next(self):

        if self.i < len(self.items):

            r = self.items[self.i]

            self.i += 1

            return(r)

        else:

            raise(StopIteration)

# This is a big ol' work in progress
class cmdLine(QtGui.QTextEdit):

    def __init__(self):

        super(cmdLine, self).__init__()

        self.currentLine = None

    def keyPressEvent(self, event):

        print(event.key())

        if event.key() in [16777234, 16777235, 16777236, 16777237]:

            return

        if event.key() == 16777220:

            print('bob')
            #self.insertPlainText('\n')

        super(cmdLine, self).keyPressEvent(event)

    def updateLog(self, text):

        self.insertPlainText(">> " + text + '\n')

        self.moveCursor(QtGui.QTextCursor.End)        
            
class preview(QtGui.QGraphicsView):

    def __init__(self):

        super(preview, self).__init__()

        scene = QtGui.QGraphicsScene()

        self.setScene(scene)

        self.photo = None
        
class canvas(QtGui.QGraphicsView):

    #Canvas is an alteration of the basic QGraphicsView. It incorperates the following functionality:

        #Drawing rectangles based on user input (code based on StackOverflow answer provided by user "yurisnm" on the thread:

            #"https://stackoverflow.com/questions/44193227/pyqt5-how-can-i-draw-inside-existing-qgraphicsview"

        #Pan and Zoom (zoom code based on StackOverflow answer provided by user "ekhumoro" on the thread:

            #"https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview"
        
    def __init__(self):

        super(canvas, self).__init__()

        scene = QtGui.QGraphicsScene()
        
        self.setScene(scene) 

        self.setMouseTracking(True)      

        self.zoom = 0

        self.empty = True

        self.photo = None

        self.patchsize = None

        self.Type = None

        self.graphicsItems = []

        self.points = []

        self.patches = []

        self.patch_dataframe = pd.DataFrame(columns=['correlation','dissimilarity','homogeneity','energy','contrast','asm', 'entropy'])

    def get_parent(self):

        return(self.parent().parent().parent().parent().parent().parent())

    def getIndex(self):

        return(self.get_parent().activeRasterIndex)

    def getCorners(self):

        cornerX = self.get_parent().rasterCoords[self.getIndex() - 1][0]

        cornerY = self.get_parent().rasterCoords[self.getIndex() - 1][1]

        return((cornerX, cornerY))

    def getGreyLevCoMatrix(self, patch):

        return(greycomatrix(patch, [5], [0], 256, symmetric = True, normed = True))

    # Might somehow incorperate edge detection in order to deliniate the raster
    # into separate texture areas before beginning classification
    def getEdges(self, patch):

        return(canny(patch, sigma=3))

    # Core algorithm for the classification of a backscatter raster image.
    def moving_patch(self):

        self.get_parent().cmdLine.updateLog("Beginning classification. This may take a while...")
        
        arr = self.get_parent().rasterArray[self.getIndex() - 1]
        size = int(self.get_parent().patchSpinBox.value())
        shp = arr.shape

        cornerX = self.getCorners()[0]
        cornerY = self.getCorners()[1]

        correlation = None
        dissimilarity = None
        homogeneity = None
        energy = None
        contrast = None
        asm = None
        entropy = None
        rugosity = None

        colorVeg = QtGui.QColor(204, 255, 204, 100)
        colorSed = QtGui.QColor(252, 199, 16, 100)
        colorRoc = QtGui.QColor(164, 0, 4, 100)

        #Load in the sql database. We will need this for RF classification
        trained_set = self.get_parent().sql_to_df()

        # Set up train and test labels/features       
        train_labels = pd.Series(trained_set['type'], dtype='category').cat.codes.values

        #print(train_labels)
        
        train_labels = np.array(train_labels)
        
        train_features = trained_set.drop('type', axis=1)
        train_features = train_features.drop('correlation', axis=1)
        train_features = train_features.drop('subtype', axis=1)
        
        feature_list = list(train_features.columns)
        
        train_features = np.array(train_features)

        test_label = [0]

        # Build the random forest classifier
        rf = RandomForestClassifier(n_estimators = 200, n_jobs = 2)

        # Train the classifier
        rf.fit(train_features, train_labels)

        # Dictionary for referencing classified types

        classified_types = {2:"Vegetation", 1:"Sediment", 0:"Rocks"}

        if len(self.get_parent().patches) > 0:

            self.get_parent().cmdLine.updateLog("Training patches still exist in memory. Click the 'Clear' button then try again.")

            return

        for x in range(0, int(shp[1] - size), size):

            #self.get_parent().cmdLine.updateLog('Classifying column ' + str(x) + ' of ' + str(shp[1]))

            for y in range(0, int(shp[0] - size), size):
                
                newPatch = patch(self.get_parent(), cornerX + x,cornerY + y, size, 'unknown', 'unknown', self.getIndex(), self.getCorners()[0],
                                 self.getCorners()[1], self.get_parent().rasterArray[self.getIndex() - 1])

                newPatchArray = newPatch.get_patch()

                glcm = self.getGreyLevCoMatrix(newPatchArray)

                if newPatch.size != 0:
                
                    correlation = greycoprops(glcm, 'correlation')[0,0]
                    dissimilarity = greycoprops(glcm, 'dissimilarity')[0,0]
                    homogeneity = greycoprops(glcm, 'homogeneity')[0,0]
                    contrast = greycoprops(glcm, 'contrast')[0,0]
                    energy = greycoprops(glcm, 'energy')[0,0]
                    asm = greycoprops(glcm, 'ASM')[0,0]
                    entropy = newPatch.get_entropy()
                    rugosity = newPatch.get_rugosity()

                    if rugosity > 10:

                        pass

                    else:

                        if correlation == 1 and homogeneity == 1 and homogeneity == 1:

                            pass

                        else:

                            # This is the feature to be classified
                            test_features = [[dissimilarity, homogeneity, contrast, energy, asm, entropy, rugosity]] # Just removed correlation from the list
                        
                            rfPredict = rf.predict(test_features)
                
                            rectItem = QtGui.QGraphicsRectItem(0,0, size, size)

                            if rfPredict[0] == 2:

                                rectItem.setBrush(QtGui.QBrush(colorVeg))

                            elif rfPredict[0] == 1:

                                rectItem.setBrush(QtGui.QBrush(colorSed))

                            else:

                                rectItem.setBrush(QtGui.QBrush(colorRoc))

                            rectItem.setPos(cornerX + x, cornerY + y)
                        
                            self.scene().addItem(rectItem)

        self.get_parent().glcmPropsToDb()

    def wheelEvent(self, event):

        if not self.empty:

            if event.delta() > 0:

                factor = 1.25
                
                self.zoom += 1

            else:

                factor = 0.8
                
                self.zoom -= 1

            self.scale(factor, factor)           

    def mouseMoveEvent(self, event):

        coords = self.mapToScene(event.x(), event.y())

        self.get_parent().xLabel.setText(str(coords.x()))
        
        self.get_parent().yLabel.setText(str(coords.y()))

        super(canvas, self).mouseMoveEvent(event)

    def drawPoint(self, x, y):
        #self.points = []
        
        pointItem = QtGui.QGraphicsTextItem()
        pointItem.setText('+')
        pointItem.setPos(x, y)
        self.points.append(pointItem)
        self.scene().addItem(self.points[-1])
        
    def mousePressEvent(self, event):

        if event.button() == 4 and self.get_parent().patchSelectionStateBox.currentText() == "Enabled":

            if self.get_parent().typeBox.currentText == "Unknown" and self.get_parent().subtypeBox.currentText() != "Unknown":

                print("If the general type is not known, how could the subtype be known?")
                return

            ####### Set coordinates and draw stuff#######
            coords = self.mapToScene(event.x(), event.y())
            
            self.patchsize = self.get_parent().patchSpinBox.value()
            
            self.rectItem = QtGui.QGraphicsRectItem(0, 0, self.patchsize, self.patchsize)
        
            self.rectItem.setPos(coords.x() - self.rectItem.boundingRect().width() / 2.0,
                                 coords.y() - self.rectItem.boundingRect().height() / 2.0)

            self.graphicsItems.append(self.rectItem)
        
            self.scene().addItem(self.graphicsItems[-1])

            # Retrieve the current seabed type being identified
            self.Type = self.get_parent().typeBox.currentText()
            self.subtype = self.get_parent().subtypeBox.currentText()

            #Debug some stuff
            #print(type(self.type))
            #print(type(self.getIndex()))
            #print(type(self.getCorners()[0]))
            #print(type(self.getCorners()[1]))
            #print(type(self.get_parent().rasterArray[self.getIndex() - 1]))

            # Create new patch variable based on user selected coordinates
            newPatch = patch(self.get_parent(), coords.x() - self.rectItem.boundingRect().width() / 2.0,
                             coords.y() - self.rectItem.boundingRect().height() / 2.0, self.patchsize, self.Type, self.subtype,
                             self.getIndex(), self.getCorners()[0], self.getCorners()[1],
                             self.get_parent().rasterArray[self.getIndex() - 1])

            # Append a new patch object to the class patch variable
            self.patches.append(newPatch)

            #print(np.var(newPatch.get_rugosity()))
            #area = newPatch.get_surface_area(newPatch.bathyPatch)
            #planarArea = newPatch.get_surface_area(newPatch.best_fit_plane(newPatch.bathyPatch))

            i = newPatch.get_pixmap()
            
            self.get_parent().drawPreview(i)
            
        super(canvas, self).mousePressEvent(event)

    def clearPatches(self):

        if len(self.graphicsItems) == 0:

            return

        self.get_parent().patches = []

        # Remove each graphics item from the scene
        for patches in self.graphicsItems:

            self.scene().removeItem(patches)
            
        # Clear the list containing the graphics items
        self.graphicsItems = []
        self.patches = []

        self.get_parent().glcm_dataframe = self.get_parent().glcm_dataframe[0:0]
        self.get_parent().patches = []
        self.get_parent().veg_patches = []
        self.get_parent().roc_patches = []
        self.get_parent().sed_patches = []

    def commitPatches(self):

        if len(self.patches) == 0:

            return

        # Save the selected patches to the MainWindow patch list.
        for x in self.patches:

            if x not in self.get_parent().patches:

                self.get_parent().patches.append(x)

        self.get_parent().glcmPropsToDb()

        confirmCommit = QtGui.QMessageBox.question(self, "Continue?","Do you wish to commit the selected patches to the SQL Database as well? This can only be undone through the SQL Command Line.", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

        if confirmCommit == QtGui.QMessageBox.Yes:

            self.get_parent().classify_to_sql(self.get_parent().glcm_dataframe)


class patch(object):

    def __init__(self, parentClass, x, y, size, Type, subtype, index, cornerX, cornerY, rasterArray):

        #Note that the patches coordinates will be that of the scene.
        #Need to convert to local image coordinates for array manipulation.
        #Subtract the rasterCoords from patch coords to get back to local system. #DONE
        self.x = x
        self.y = y

        self.rasterArray = rasterArray

        self.index = index

        self.cornerX = cornerX
        self.cornerY = cornerY
        
        self.size = size
        self.Type = Type
        self.subtype = subtype

        self.patch = None
        self.bathyPatch = None

        self.parentClass = parentClass

        #self.image = self.parent().parent().parent().rasterArray[self.index]

    # Convert the active patch to a pixmap for display on the GUI
    def get_pixmap(self):

        patch = self.get_patch()

        arr_Image = Image.fromarray(patch)

        arr_Image = QtGui.QImage(ImageQt.ImageQt(arr_Image))

        qPixMap = QtGui.QPixmap.fromImage(arr_Image)

        return(qPixMap)
    
    # Retieve bathymetry values which correspond with the current patch
    def get_patch_bathy(self):

        # The coordinates for the northwestern corner of the bathymetry raster
        bathyX = self.parentClass.bathyCoords[0]
        bathyY = self.parentClass.bathyCoords[1]

        # Get offset coordinates so that the bathy matches the backscatter
        offsetX = bathyX - self.cornerX
        offsetY = bathyY - self.cornerY

        # Need to get local coordinates of the patch and offset them by the values above
        x = self.get_local_coords()[0] + offsetX
        y = self.get_local_coords()[1] + offsetY

        # Create a patch of bathymetry data which corresponds to the backscatter data.
        patch_bathy = self.parentClass.bathy[int(y): int(y) + int(self.size), int(x): int(x) + int(self.size)]
        self.bathyPatch = patch_bathy
        
        return(patch_bathy, x, y)

    # returns the length of a 3D vector
    def get_vector_length(self, vector):

        return(((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2) ** 0.5))

    # returns the dot product of a 3d vector
    def dot_product_3d(self, u, v):

        return(u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
        
    # Get angle between a set of 2 3D vectors
    def get_vector_props(self, u, v):

        dotVectors = self.dot_product_3d(u, v)
        lengthU = self.get_vector_length(u)
        lengthV = self.get_vector_length(v)

        angle = degrees(acos(dotVectors / (lengthU * lengthV)))

        return(angle, lengthU, lengthV)
                   
    def get_surface_area(self, inputArr):

        #arr = self.bathyPatch

        total_area = 0

        skip_count = 0

        gridSize = self.parentClass.bathyGridSize

        arr = inputArr.astype(float)

        for x in range(0, arr.shape[1] - 1, 2):

            for y in range(0, arr.shape[0] - 1, 2):

                currentBlock = arr[int(y): int(y) + 2, int(x): int(x) + 2]

                # Get the angles and sides from the vectors which make up the active block.
                v1 = self.get_vector_props((0,gridSize,currentBlock[0][1] - currentBlock[0][0]),(gridSize, 0, currentBlock[1][0] - currentBlock[0][0]))
                v2 = self.get_vector_props((0,gridSize,currentBlock[0][0] - currentBlock[0][1]),(gridSize, 0, currentBlock[1][1] - currentBlock[0][1]))
                v3 = self.get_vector_props((gridSize,0,currentBlock[0][1] - currentBlock[1][1]),(0, gridSize, currentBlock[1][0] - currentBlock[1][1]))
                v4 = self.get_vector_props((gridSize,0,currentBlock[0][0] - currentBlock[1][0]),(0, gridSize, currentBlock[0][0] - currentBlock[1][0]))

                # Yeah, I know storing the lengths with the angles in a tuple would end up being slightly confusing, but alas, earwax.
                # Also, I capitolized the L's because otherwise they look like the number 1.
                L1 = v1[1]
                L2 = v2[2]
                L3 = v3[2]
                L4 = v4[1]

                # a1 and a3 are opposite angles
                # angle a1 connects sides L1 and L4
                # angle a2 connects sides L1 and L2
                # angle a3 connects sides L2 and L3
                # andle a4 connects sides L4 and L3
                a1 = v1[0]
                a2 = v2[0]
                a3 = v3[0]
                a4 = v4[0]

                # Calculate the area of the quadrilateral formed by the four vectors.                    
                area = (0.5 * (L1 * L4) * sin(radians(a1))) + (0.5 * (L2 * L3) * sin(radians(a3)))

                if area < 100000 and area > -100000:

                    total_area += area

                else:

                    skip_count += 1
                    
                #print(str(area) + ' = ' + str(0.5) + ' * (' + str(L1) + ' * ' + str(L4) + ') * ' + str(sin(radians(a1))) + ' + (0.5 * (' + str(L2) + ' * ' + str(L3) + ') * ' + str(sin(radians(a3))))

        #print("Skipped " + str(skip_count) + " due to missing data points.")
        #print("Total Area: " + str(total_area))
        #print("Planar Area: " + str(planar_area))
        #print("Area Ratio: " + str(area / planar_area))
        
        return(total_area)

    # interpolate between two values
    def interpolate(self, val1, val2, steps):

        interp = []

        slope = (val2 - val1) / steps

        i = val1
        
        for x in range(int(steps) + 1):

            i += slope

            interp.append(i)

        return(interp)

    # Returns a plane which matches the patch at all four corners.
    def best_fit_plane(self, arr):

        arr = arr.astype(float)

        # Select the first column of the array. Interpolate from the first value to the final value.
        arr[0:,0] = self.interpolate(float(arr[0:,0][0]), float(arr[0:,0][-1]), len(arr[0:,0]) - 1)
        
        arr[1:,0] = self.interpolate(float(arr[1:,0][0]), float(arr[1:,0][-1]), len(arr[1:,0]) - 1)
        
        # for x in the range of the length of the first row, interpolate each row from left to right.
        for x in range(len(arr[0])):

            arr[x] = self.interpolate(float(arr[x][0]), float(arr[x][-1]), len(arr[x]) - 1)

        return(arr)

    def get_rugosity(self):

        patch = self.get_patch_bathy()[0]
        
        area = self.get_surface_area(patch)
        
        planar = self.get_surface_area(self.best_fit_plane(patch))

        return(area / planar)

    def get_Active_Raster_Index(self):

        return(self.index)

    def get_geo_coords(self):

        return((self.x, self.y))

    def get_size(self):

        return(self.size)

    def get_subtype(self):

        return(self.subtype)

    def get_type(self):

        return(self.Type)

    def get_patch(self):

        x = self.get_local_coords()[0]
        y = self.get_local_coords()[1]

        index = self.get_Active_Raster_Index() - 1
        
        patch = []

        patch = self.rasterArray[int(y): int(y) + int(self.size), int(x): int(x) + int(self.size)]

        self.patch = patch
        
        return(patch)

    def get_local_coords(self):

        i = self.get_Active_Raster_Index() - 1

        x = self.x - self.cornerX
        
        y = self.y - self.cornerY

        return((x, y))

    def get_entropy(self):

        i = np.squeeze(self.get_glcm())

        return(-np.sum(i * np.log2(i + (i == 0))))

    def get_glcm(self):

        return(greycomatrix(self.get_patch(), [5], [0], 256, symmetric = True, normed = True))

app = QtGui.QApplication(sys.argv)

form = classiPy2_Form()

form.show()

app.exec_()
