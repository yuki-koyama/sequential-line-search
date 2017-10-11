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

#if 0
    static void drawFloor(int n, Y::Vector3d color1, Y::Vector3d color2);

    static Y::Vector3d unproject(const Y::Vector2d& screenPos);
    static Y::Vector2d project(const Y::Vector3d& worldPos);

    // OpenSCAD (3d)
    static void drawSphere(double r, Y::Vector3d p, int res);
    static void drawSphere(double r, int res);
    static void drawCylinder(double r, double h, int res = 60, bool center = true);
    static void drawCylinder(double r1, double r2, double h, int res, bool center = true);
    static void drawCylinder(double r, Y::Vector3d p1, Y::Vector3d p2, int res);
    static void drawCube(double size, bool center = true);
    static void drawCube(double x, double y, double z, bool center = true);
    static void drawCube(Y::Vector3d translate, double x, double y, double z, bool center = true);
    // OpenSCAD (2d to 3d)
    static void drawConvexLinearExtrusion(double h, std::vector<Y::Vector2d> x, bool center = true);
    static void drawPrism(double s, double angle1, double angle2, double h);
    // OpenSCAD (2d)
    static std::vector<Y::Vector2d> square2d(double w, double h, bool center = true);
    static std::vector<Y::Vector2d> translate2d(std::vector<Y::Vector2d> points, double x, double y);

    static void drawCurrentFrame(double length, double width = 3.0);

    // originally by Kenshi Takayama
    template <typename Scalar>
    static Y::Matrix<Scalar, 3, 3> rotationFromAxisAngle(Y::Vector<Scalar, 3>& axis, Scalar angle) {
        using Y::Matrix;
        Matrix<Scalar, 3, 3> m;
        axis.normalize();
        Scalar c = cos(angle);
        Scalar s = sin(angle);
        Scalar omc = 1.0 - c;
        m(0, 0) = c + axis[0] * axis[0] * omc;
        m(1, 1) = c + axis[1] * axis[1] * omc;
        m(2, 2) = c + axis[2] * axis[2] * omc;

        Scalar tmp1, tmp2;

        tmp1 = axis[0] * axis[1] * omc;
        tmp2 = axis[2] * s;
        m(0, 1) = tmp1 - tmp2;
        m(1, 0) = tmp1 + tmp2;

        tmp1 = axis[2] * axis[0] * omc;
        tmp2 = axis[1] * s;
        m(2, 0) = tmp1 - tmp2;
        m(0, 2) = tmp1 + tmp2;

        tmp1 = axis[1] * axis[2] * omc;
        tmp2 = axis[0] * s;
        m(1, 2) = tmp1 - tmp2;
        m(2, 1) = tmp1 + tmp2;

        return m;
    }
#endif
};

#endif // DRAWUTILITY_H
