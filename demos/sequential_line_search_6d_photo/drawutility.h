#ifndef DRAWUTILITY_H
#define DRAWUTILITY_H

#include <string>
#include <vector>
#include <OpenGL/gl.h>

class DrawUtility
{
public:
    static int  loadShader(std::string vtxShdName, std::string frgShdName, GLuint *lpProg);
    static void printShaderLog(GLuint shader);
    static void printProgramInfoLog(GLuint program);
};

#endif // DRAWUTILITY_H
