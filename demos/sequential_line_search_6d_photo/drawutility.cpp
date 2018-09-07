#include "drawutility.h"

#include <list>
#include <stdlib.h>

using std::string;
using std::vector;

namespace
{
extern int _loadShader(GLuint shader, string shdName);
}

int DrawUtility::loadShader(string vtxShdName, string frgShdName, GLuint *lpProg)
{
    GLuint vtxShader;
    GLuint frgShader;
    GLuint prog;
    GLint linked;

    vtxShader = glCreateShader(GL_VERTEX_SHADER);
    frgShader = glCreateShader(GL_FRAGMENT_SHADER);

    if (_loadShader(vtxShader, vtxShdName) < 0) {
        return -1;
    }

    if (_loadShader(frgShader, frgShdName) < 0) {
        return -1;
    }

    prog = glCreateProgram();

    glAttachShader(prog, vtxShader);
    glAttachShader(prog, frgShader);

    glDeleteShader(vtxShader);
    glDeleteShader(frgShader);

    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);

    if (linked == GL_FALSE) {
        fprintf(stderr, "Link error of %s & %s!!\n", vtxShdName.c_str(), frgShdName.c_str());
        printProgramInfoLog(prog);
        return -1;
    }

    *lpProg = prog;

    return 0;
}

void DrawUtility::printShaderLog(GLuint shader) {
    int logSize;
    int length;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH , &logSize);

    GLchar *log = (GLchar *)malloc(sizeof(GLchar) * logSize);

    if (logSize > 1) {
        glGetShaderInfoLog(shader, logSize, &length, log);
        fprintf(stderr, "Shader Info Log\n%s\n", log);
    }

    free(log);
}

void DrawUtility::printProgramInfoLog(GLuint program) {
    GLsizei bufSize;

    glGetProgramiv(program, GL_INFO_LOG_LENGTH , &bufSize);

    if (bufSize > 1) {
        GLchar *infoLog;

        infoLog = (GLchar *)malloc(bufSize);
        if (infoLog != NULL) {
            GLsizei length;

            glGetProgramInfoLog(program, bufSize, &length, infoLog);
            fprintf(stderr, "Program Info Log:\n%s\n", infoLog);
        } else {
            fprintf(stderr, "Could not allocate InfoLog buffer.\n");
        }
        free(infoLog);
    }
}

namespace
{
int _loadShader(GLuint shader, string shdName) {
    FILE *fp;
    void *buf;
    int size;
    GLint compiled;

    if ((fp = fopen(shdName.c_str(), "rb")) == NULL) {
        fprintf(stderr, "%s is not found!!\n", shdName.c_str());
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    size = (int)ftell(fp);

    if ((buf = (void *)malloc(size + 1)) == NULL) {
        fprintf(stderr, "Memory is not enough for %s\n", shdName.c_str());
        fclose(fp);
        return -1;
    }

    fseek(fp, 0, SEEK_SET);
    fread(buf, 1, size, fp);
    ((char *)buf)[size] = '\0';

    string wholeSource((char *)buf);

    int wholeSize = (int)wholeSource.size();
    GLchar *source = (GLchar *)malloc(wholeSize + 1);
    memcpy(source, wholeSource.c_str(), wholeSize + 1);
    source[wholeSize] = '\0';

    //	glShaderSource(shader, 1, (const GLchar **)&buf, &size);
    glShaderSource(shader, 1, (const GLchar **)&source, &wholeSize);

    free(buf);
    free(source);
    fclose(fp);

    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

    if (compiled == GL_FALSE)
    {
        fprintf(stderr, "Compile error in %s!!\n", shdName.c_str());
        DrawUtility::printShaderLog(shader);
        return -1;
    }

    return 0;
}
}

#if 0
void DrawUtility::drawFloor(int n, Vector3d color1, Vector3d color2)
{
    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0, 1.0);
    glBegin(GL_QUADS);
    glNormal3d(0.0, 1.0, 0.0);
    for (int i = -n; i < n; ++ i) {
        for (int j = -n; j < n; ++ j) {
            if ((i + j) % 2 == 0) {
                glColor3dv(color1.val);
            } else {
                glColor3dv(color2.val);
            }
            double x1 = static_cast<double>((i + 0) * n);
            double x2 = static_cast<double>((i + 1) * n);
            double z1 = static_cast<double>((j + 0) * n);
            double z2 = static_cast<double>((j + 1) * n);
            glVertex3d(x2, 0.0, z2);
            glVertex3d(x2, 0.0, z1);
            glVertex3d(x1, 0.0, z1);
            glVertex3d(x1, 0.0, z2);
        }
    }
    glEnd();
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
}

Vector3d DrawUtility::unproject(const Vector2d &screenPos)
{
    int vp[4];
    double model[16], proj[16];
    glGetIntegerv(GL_VIEWPORT, vp);
    glGetDoublev(GL_MODELVIEW_MATRIX,  model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    Vector3d worldPos;
    gluUnProject(screenPos[0], vp[3] - 1 - screenPos[1], 1.0, model, proj, vp, &worldPos[0], &worldPos[1], &worldPos[2]);
    return worldPos;
}

Vector2d DrawUtility::project(const Vector3d &worldPos)
{
    int vp[4];
    double model[16], proj[16];
    glGetIntegerv(GL_VIEWPORT, vp);
    glGetDoublev(GL_MODELVIEW_MATRIX,  model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    Vector3d screenPos;
    gluProject(worldPos.x, worldPos.y, worldPos.z, model, proj, vp, &screenPos[0], &screenPos[1], &screenPos[2]);
    return Vector2d(screenPos.x, vp[3] - 1 - screenPos.y);
}

void DrawUtility::drawSphere(double r, Vector3d p, int res)
{
    glPushMatrix();
    glTranslated(p.x, p.y, p.z);
    drawSphere(r, res);
    glPopMatrix();
}

void DrawUtility::drawSphere(double r, int res)
{
    glutSolidSphere(r, res, res);
}

void DrawUtility::drawCylinder(double r, double h, int res, bool center)
{
    drawCylinder(r, r, h, res, center);
}

void DrawUtility::drawCylinder(double r1, double r2, double h, int res, bool center)
{
    glPushMatrix();
    if (center) glTranslatef(0.0f, 0.0f, - static_cast<GLfloat>(h / 2.0));
    GLUquadricObj* qobj = gluNewQuadric();
    gluCylinder(qobj, r1, r2, h, res, 1);
    glScalef(-1.0f, 1.0f, -1.0f);
    gluDisk(qobj, 0.0, r1, res, 1);
    glScalef(-1.0f, 1.0f, -1.0f);
    glTranslatef(0.0f, 0.0f, static_cast<GLfloat>(h));
    gluDisk(qobj, 0.0, r2, res, 1);
    gluDeleteQuadric(qobj);
    glPopMatrix();
}

void DrawUtility::drawCylinder(double r, Vector3d p1, Vector3d p2, int res)
{
    Vector3d   t   = p2 - p1;
    Vector3d   t0  = Vector3d(0.0, 0.0, 1.0);
    double     c   = t.dot_product(t0) / t.length();
    double     q   = acos(c);
    Vector3d   ax  = t0.cross_product(t).normalized();
    Matrix4x4d rot = Matrix4x4d::rotationFromAxisAngle(ax, q);
    double     h   = t.length();

    if (fabs(c - 1.0) < 1e-16) rot = Matrix4x4d::identity();

    glPushMatrix();
    glTranslated(p1.x, p1.y, p1.z);
    glMultMatrixd(rot.transpose().values_);
    drawCylinder(r, h, res, false);
    glPopMatrix();
}

void DrawUtility::drawCube(double size, bool center)
{
    assert (center);

    glutSolidCube(size);
}

void DrawUtility::drawCube(double x, double y, double z, bool center)
{
    assert (center);
    glPushMatrix();
    glScaled(x, y, z);
    glutSolidCube(1.0);
    glPopMatrix();
}

void DrawUtility::drawCube(Vector3d translate, double x, double y, double z, bool center)
{
    assert (center);
    glPushMatrix();
    glTranslated(translate.x, translate.y, translate.z);
    glScaled(x, y, z);
    glutSolidCube(1.0);
    glPopMatrix();
}

void DrawUtility::drawConvexLinearExtrusion(double h, vector<Vector2d> x, bool center)
{
    double zt = center ? h / 2.0 : h;
    double zb = center ? - h / 2.0 : 0.0;

    int n = x.size();

    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0.0, 0.0, 1.0);
    for (unsigned i = 0; i < x.size(); ++ i) {
        Vector2d p = x[i];
        glVertex3d(p[0], p[1], zt);
    }
    glEnd();

    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0.0, 0.0, -1.0);
    for (int i = x.size() - 1; i >= 0; -- i) {
        Vector2d p = x[i];
        glVertex3d(p[0], p[1], zb);
    }
    glEnd();

    glBegin(GL_QUADS);
    for (int i = 0; i < n; ++ i) {
        Vector2d& p1 = x[i];
        Vector2d& p2 = x[(i + 1) % n];
        Vector3d p1t(p1[0], p1[1], zt);
        Vector3d p1b(p1[0], p1[1], zb);
        Vector3d p2t(p2[0], p2[1], zt);
        Vector3d p2b(p2[0], p2[1], zb);
        Vector3d n = (p1b - p1t).cross_product(p2t - p1t).normalized();
        glNormal3dv(n.ptr());
        glVertex3dv(p2t.ptr());
        glVertex3dv(p1t.ptr());
        glVertex3dv(p1b.ptr());
        glVertex3dv(p2b.ptr());
    }
    glEnd();
}

void DrawUtility::drawPrism(double s, double angle1, double angle2, double h)
{
    vector<Vector2d> x;
    x.push_back(Vector2d(0.0, 0.0));
    x.push_back(Vector2d(s * cos(angle1), s * sin(angle1)));
    x.push_back(Vector2d(s * cos(angle2), s * sin(angle2)));
    drawConvexLinearExtrusion(h, x);
}

void DrawUtility::drawCurrentFrame(double length, double width)
{
    double l = length;
    double m = 0.0;
    glDisable(GL_LIGHTING);
    glLineWidth(width);
    glBegin(GL_LINES);
    glColor3d (0.9, 0.0, 0.0);
    glVertex3d(  m, 0.0, 0.0);
    glVertex3d(  l, 0.0, 0.0);
    glColor3d (0.0, 0.9, 0.0);
    glVertex3d(0.0,   m, 0.0);
    glVertex3d(0.0,   l, 0.0);
    glColor3d (0.0, 0.0, 0.9);
    glVertex3d(0.0, 0.0,   m);
    glVertex3d(0.0, 0.0,   l);
    glEnd();
    glEnable(GL_LIGHTING);
}

vector<Vector2d> DrawUtility::square2d(double w, double h, bool center)
{
    assert (center);

    double w2 = w / 2.0;
    double h2 = h / 2.0;

    vector<Vector2d> sq;
    sq.push_back(Vector2d(w2, h2));
    sq.push_back(Vector2d(-w2, h2));
    sq.push_back(Vector2d(-w2, -h2));
    sq.push_back(Vector2d(w2, -h2));
    return sq;
}

vector<Vector2d> DrawUtility::translate2d(vector<Vector2d> points, double x, double y)
{
    for (Vector2d& p : points) {
        p[0] += x;
        p[1] += y;
    }
    return points;
}
#endif
