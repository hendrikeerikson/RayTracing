import pyopencl as cl

from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget

from OpenGL.GL import *

import numpy as np
import sys

import os
from AssistingFuntions import *
from Constants import *

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


class GLPlotWidget (QGLWidget):
    width, height = 400, 300

    def initialize_buffer(self):
        # openCL initialization
        self.ctx, self.queue, self.device = cl_init()
        self.program = cl.Program(self.ctx, read_kernel("colorize.cls")).build()

        mf = cl.mem_flags

        # matrix that will contain the returned pixel data
        self.screen_matrix = np.zeros((self.height, self.width), dtype=np.uint32)
        self.matrix_shape = (self.height, self.width)  # shape of the pixel data matrix
        self.screen_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.screen_matrix.nbytes)

        self.create_camera_buffers()
        self.queue.finish()

    def create_camera_buffers(self):
        # returns a struct containing the requiered vectors to calculate the initial ray direction
        cam_struct = create_camera_struct(self.width, self.height, self.device)
        mf = cl.mem_flags

        self.camera_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cam_struct)

    def execute(self):
        # run the program
        self.program.colorize(self.queue, (self.matrix_shape[1], self.matrix_shape[0]), None, self.screen_buffer,
                              self.camera_buffer)

        # copy the pixel data from the buffer to the screen matrix
        cl.enqueue_read_buffer(self.queue, self.screen_buffer, self.screen_matrix).wait()
        self.queue.finish()

        # make a byte array from the screen matrix to be converted into a texture buffer
        self.byte_array = self.screen_matrix.tobytes()

        self.texture = texture_from_array(self.byte_array, self.matrix_shape)

    def update_buffer(self):
        self.execute()
        glFlush()

    # called once before exicution
    def initializeGL(self):
        self.initialize_buffer()

        glClearColor(0.3, 0.3, 0.3, 0)
        self.update_buffer()

    # called every frame
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glColor3f(1, 1, 1)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1);     glVertex2f(-1, 1)
        glTexCoord2f(0, 0);     glVertex2f(-1, -1)
        glTexCoord2f(1, 0);     glVertex2f(1, -1)
        glTexCoord2f(1, 1);     glVertex2f(1, 1)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glFlush()

    # called when the window is resized
    def resizeGL(self, width, heigth):
        self.width = width
        self.height = heigth

        glViewport(0, 0, width, heigth)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, 1, -1, -1, 1)


if __name__ == '__main__':
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            self.widget = GLPlotWidget()
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()

    app = QtGui.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec_()
