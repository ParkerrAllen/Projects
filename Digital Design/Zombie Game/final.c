#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<math.h>
#ifdef USEGLEW
#include<GL/glew.h>
#endif
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include<GLUT/glut.h>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#else
#include<GL/glut.h>
#endif
#include"CSCIx229.h"

// State variables
int t=0,p=0,psp=2,axs=1;
double fov=55,a=1.0,d=15.0;
int winW=800,winH=800;

enum GameState {RUNNING, GAME_OVER, GAME_WIN};
enum GameState cGS=RUNNING;

// Timer variables
double sTime=0;
double eTime=0;
double tLimit=180.0; // 3 minutes
// Airplane variables
double ap_ang=0,ap_r1=30,ap_r2=15;

// Player state
double cX=0.0,cY=1.8, cZ=8.0,cYaw=0.0,cPitch=0.0;
float playerHealth=100.0;
// Moon variables
double m_ang=0,m_r=20;

// Lighting parameters
float amb=10,diff=100,spec=0,shiny=0,ylt=5.0;
int l_move=0;
float l_ang=0;
double sp_ang=0;

#define PLAYER_R 0.5
#define HOUSE_R 1.5
#define ROCK_R 0.7
#define TREE_R 0.6

// Define zombie positions
#define NUM_ZOMBIES 50
double asa = 0.0;
double lsa= 0.0;
double zspeed = 0.01;
float lHit = 0.0; 
double zX[NUM_ZOMBIES]={5,-5}, zY[NUM_ZOMBIES]={0, 0},zZ[NUM_ZOMBIES]={5.0, -5};
int CZombies=2;
int zombDam=5;
#define NUM_STREET_LIGHTS 3
double stLtPos[NUM_STREET_LIGHTS][3]={
    {2,0,2},
    {-3,0,-2.0},
    {5,0,-5}
};

#define NUM_BARRICADES 3
double barPos[NUM_BARRICADES][3]={
    {3,0,3},
    {-4,0,-2},
    {7,0,-7}
};
unsigned int tex[17];
#define LEN 8192
#define NUM_HOUSES 3
double hPos[NUM_HOUSES][3]={
    {0,0,0},
    {-8,0,5},
    {7,0,-9}
};

#define NUM_ROCKS 3
double rPos[NUM_ROCKS][3]={
    {2,0,-3},
    {-6,0,4},
    {9,0,1}
};
#define NUM_HUMANS 3
double huPos[NUM_HUMANS][3]={
    {2.5,0,0},
    {-5.5,0,5},
    {9,0,-9}
};

void Print(const char* fmt,...){
    char buf[LEN],*ch=buf;
    va_list args;
    va_start(args,fmt);
    vsnprintf(buf,LEN,fmt,args);
    va_end(args);
    while(*ch)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,*ch++);
}

void renderTextOnScreen(){
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW,0, winH);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    glColor3f(1, 1, 1); 
    int textWidth = 300;
    glRasterPos2i(winW- textWidth, winH-20); 
    Print("Mode: %s | Control: %s",
          psp==0?"Orthogonal" : (psp==1 ? "Perspective" : "First-Person"),
          l_move ? "Light" : "Camera");
    double remTime=tLimit - eTime;
    glRasterPos2i(winW-textWidth, winH-40);
    Print("Time Remaining: %.2f seconds", fmax(remTime, 0));
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}




// Load and bind textures
void loadTextures(){
    tex[0]=LoadTexBMP("image3.bmp");
    tex[1]=LoadTexBMP("image4.bmp");
    tex[2]=LoadTexBMP("image2.bmp");
    tex[3]=LoadTexBMP("image1.bmp");
    tex[4]=LoadTexBMP("image5.bmp");
    tex[5]=LoadTexBMP("image6.bmp");
    tex[6]=LoadTexBMP("image7.bmp");
    tex[7]=LoadTexBMP("image8.bmp");
    tex[8]=LoadTexBMP("image9.bmp");
    tex[9]=LoadTexBMP("skin.bmp");
    tex[10]=LoadTexBMP("clothes.bmp");
    tex[11]=LoadTexBMP("bark.bmp");
    tex[12]=LoadTexBMP("leaves.bmp");
    tex[13]=LoadTexBMP("bar.bmp");
    tex[14]=LoadTexBMP("rock.bmp");
    tex[15]=LoadTexBMP("metal.bmp");
    tex[16]=LoadTexBMP("skyscraper.bmp");
}

#define NUM_SKYSCRAPERS 25
double skyPos[NUM_SKYSCRAPERS][6];

void InitSkyPos(){
    srand(0);
    for(int i=0;i<NUM_SKYSCRAPERS;i++){
        double x =((rand()% 100)- 50);
        double z =((rand() % 100) -50);
        double width =4.0 +(rand()%4); 
        double height = 20.0+(rand()% 20);
        double depth = width;
        skyPos[i][0]= x;
        skyPos[i][1]= 0.0; 
        skyPos[i][2] =z;
        skyPos[i][3]=width;
        skyPos[i][4]=height;
        skyPos[i][5]=depth;
    }
}
// get ground height

float getGH(double x,double z) {
    float maxH=2.0f;
    return sinf(x*0.2f)*cosf(z*0.2f)*maxH;
}

void updateHumanPos() {
    for(int i=0;i<NUM_HUMANS;i++) {
        double x=huPos[i][0],z=huPos[i][2];
        huPos[i][1]=getGH(x,z);
    }
}

void updateHousePos() {
    for(int i=0;i<NUM_HOUSES;i++) {
        double x=hPos[i][0],z=hPos[i][2];
        hPos[i][1]=getGH(x,z);
    }
}

void updateRockPos() {
    for(int i=0;i<NUM_ROCKS;i++) {
        double x=rPos[i][0],z=rPos[i][2];
        rPos[i][1]=getGH(x,z);
    }
}

void cylinder(double x,double y,double z,double r,double h,int sl){
    const double as=360.0/sl;
    glPushMatrix();
    glTranslated(x,y,z);
    glEnable(GL_TEXTURE_2D);
    // Draw sides
    glBegin(GL_QUAD_STRIP);
    for(double ang=0;ang<=360;ang+=as){
        double rad=ang*M_PI/180.0;
        double s=ang/360.0;
        glNormal3f(cos(rad),0.0,sin(rad));
        glTexCoord2f(s,0.0);
        glVertex3f(r*cos(rad),0.0,r*sin(rad));
        glTexCoord2f(s,1.0);
        glVertex3f(r*cos(rad),h,r*sin(rad));
    }
    glEnd();
    // Draw top cap
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.0,1.0,0.0);
    glTexCoord2f(0,0);
    glVertex3f(0,h,0);
    for(double ang=0;ang<=360;ang+=as){
        double rad=ang*M_PI/180.0;
        double s=0.5+0.5*cos(rad);
        double t=0.5+0.5*sin(rad);
        glTexCoord2f(s,t);
        glVertex3f(r*cos(rad),h,r*sin(rad));
    }
    glEnd();
    // Draw bottom cap
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0,-1,0);
    glTexCoord2f(0.5,0.5);
    glVertex3f(0,0,0);
    for(double ang=0;ang<=360;ang+=as){
        double rad=ang*M_PI/180;
        double s=0.5+0.5*cos(rad);
        double t=0.5+0.5*sin(rad);
        glTexCoord2f(s,t);
        glVertex3f(r*cos(rad),0,r*sin(rad));
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

void updateBarPos() {
    for(int i=0;i<NUM_BARRICADES;i++) {
        double x=barPos[i][0],z=barPos[i][2];
        barPos[i][1]=getGH(x,z);
    }
}

void drawAirplane(){
    double pX=ap_r1*cos(ap_ang),pY=10.0,pZ=ap_r2*sin(ap_ang);
    double dx=-ap_r1*sin(ap_ang),dz=ap_r2*cos(ap_ang);
    double p_dir=atan2(dz,dx)*(180.0/M_PI);
    glPushMatrix();
    glTranslated(pX,pY,pZ);
    glRotated(-p_dir,0,1,0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex[6]);
    glPushMatrix();
    glScaled(3.0,0.5,0.5);
    glBegin(GL_QUADS);
    // Front face
    glNormal3f(0,0,1);
    glTexCoord2f(0,0); glVertex3f(-0.5,-0.5,0.5);
    glTexCoord2f(1,0); glVertex3f(0.5,-0.5,0.5);
    glTexCoord2f(1,1); glVertex3f(0.5,0.5,0.5);
    glTexCoord2f(0,1); glVertex3f(-0.5,0.5,0.5);
    // Back face
    glNormal3f(0,0,-1);
    glTexCoord2f(0,0); glVertex3f(0.5,-0.5,-0.5);
    glTexCoord2f(1,0); glVertex3f(-0.5,-0.5,-0.5);
    glTexCoord2f(1,1); glVertex3f(-0.5,0.5,-0.5);
    glTexCoord2f(0,1); glVertex3f(0.5,0.5,-0.5);
    // Left face
    glNormal3f(-1,0,0);
    glTexCoord2f(0,0); glVertex3f(-0.5,-0.5,-0.5);
    glTexCoord2f(1,0); glVertex3f(-0.5,-0.5,0.5);
    glTexCoord2f(1,1); glVertex3f(-0.5,0.5,0.5);
    glTexCoord2f(0,1); glVertex3f(-0.5,0.5,-0.5);
    // Right face
    glNormal3f(1,0,0);
    glTexCoord2f(0,0); glVertex3f(0.5,-0.5,0.5);
    glTexCoord2f(1,0); glVertex3f(0.5,-0.5,-0.5);
    glTexCoord2f(1,1); glVertex3f(0.5,0.5,-0.5);
    glTexCoord2f(0,1); glVertex3f(0.5,0.5,0.5);
    // Top face
    glNormal3f(0,1,0);
    glTexCoord2f(0,0); glVertex3f(-0.5,0.5,0.5);
    glTexCoord2f(1,0); glVertex3f(0.5,0.5,0.5);
    glTexCoord2f(1,1); glVertex3f(0.5,0.5,-0.5);
    glTexCoord2f(0,1); glVertex3f(-0.5,0.5,-0.5);
    // Bottom face
    glNormal3f(0,-1,0);
    glTexCoord2f(0,0); glVertex3f(-0.5,-0.5,-0.5);
    glTexCoord2f(1,0); glVertex3f(0.5,-0.5,-0.5);
    glTexCoord2f(1,1); glVertex3f(0.5,-0.5,0.5);
    glTexCoord2f(0,1); glVertex3f(-0.5,-0.5,0.5);
    glEnd();
    glPopMatrix();
    // Draw the wings
    glBindTexture(GL_TEXTURE_2D,tex[7]);
    glBegin(GL_QUADS);
    // Left wing
    glNormal3f(0,1,0);
    glTexCoord2f(0,0); glVertex3f(0,0,0.5);
    glTexCoord2f(1,0); glVertex3f(2.0,0,0.5);
    glTexCoord2f(1,1); glVertex3f(2.0,0,2.0);
    glTexCoord2f(0,1); glVertex3f(0,0,2.0);
    // Right wing
    glTexCoord2f(0,0); glVertex3f(0,0,-0.5);
    glTexCoord2f(1,0); glVertex3f(2.0,0,-0.5);
    glTexCoord2f(1,1); glVertex3f(2.0,0,-2.0);
    glTexCoord2f(0,1); glVertex3f(0,0,-2.0);
    glEnd();
    // Draw the tail fin
    glBegin(GL_TRIANGLES);
    glNormal3f(0,0,1);
    glTexCoord2f(0.5,1); glVertex3f(-1.5,0.0,0.0);
    glTexCoord2f(0,0); glVertex3f(-1.0,0.5,0.0);
    glTexCoord2f(1,0); glVertex3f(-1.0,0.0,0.0);
    glEnd();
    // Draw the propeller
    glPushMatrix();
    glTranslated(1.5,0,0);
    glRotated(sp_ang*10,1,0,0);
    glBegin(GL_QUADS);
    glNormal3f(1,0,0);
    glTexCoord2f(0,0); glVertex3f(0,-0.05,-0.5);
    glTexCoord2f(1,0); glVertex3f(0,-0.05,0.5);
    glTexCoord2f(1,1); glVertex3f(0,0.05,0.5);
    glTexCoord2f(0,1); glVertex3f(0,0.05,-0.5);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

void drawSphere(double x,double y,double z,double r){
    const int d=15;
    glPushMatrix();
    glTranslated(x,y,z);
    glScaled(r,r,r);
    for(int ph=-90;ph<90;ph+=d){
        glBegin(GL_QUAD_STRIP);
        for(int th=0;th<=360;th+=d){
            double x1=Sin(th)*Cos(ph);
            double y1=Sin(ph);
            double z1=Cos(th)*Cos(ph);
            double s=(double)th/360.0;
            double t=(double)(ph+90)/180.0;
            glNormal3d(x1,y1,z1);
            glTexCoord2d(s,t);
            glVertex3d(x1,y1, z1);
            double x2=Sin(th) *Cos(ph+d);
            double y2=Sin(ph+d);
            double z2=Cos(th)* Cos(ph+d);
            double t2=(double)(ph+d+90) /180.0;
            glNormal3d(x2,y2,z2);
            glTexCoord2d(s,t2);
            glVertex3d(x2,y2,z2);
        }
        glEnd();
    }
    glPopMatrix();
}

void drawTree(double x,double y,double z,double s){
    glPushMatrix();
    glTranslated(x,y,z);
    glScaled(s,s,s);
    // Draw trunk
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex[11]); // Bark texture
    glPushMatrix();
    glColor3f(0.55f,0.27f,0.07f);
    cylinder(0,0,0,0.2,2.0,16);
    glPopMatrix();
    // Draw leaves
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBindTexture(GL_TEXTURE_2D,tex[12]);
    glColor4f(0.0f,0.5f,0.0f,0.8f); 
    double leafPos[][3]={
        {0,2.5,0},{0.5,2.3,.2},{-0.5,2.2,-0.3},
        {.3,2.7,-0.4},{-0.4,2.6,.5},{0.0,2.8,0.0},
        {.6,2.4,-0.6},{-0.6,2.5,0.6},{0.2,2.9,0.3},
        {-0.2,2.9,-0.3},{0.0,3,0.0}
    };
    double leafSizes[]={0.8,0.6,0.7,0.5,.6,.4,.5,.5,.3,.3,.4};
    int numLeaves=sizeof(leafSizes)/sizeof(leafSizes[0]);
    for(int i=0;i<numLeaves;i++){
        glPushMatrix();
        glTranslated(leafPos[i][0],leafPos[i][1],leafPos[i][2]);
        drawSphere(0,0,0,leafSizes[i]);
        glPopMatrix();
    }
    glDisable(GL_BLEND);
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    glPopMatrix();
}
void updateSkyscraperPos(){
    for (int i = 0; i<NUM_SKYSCRAPERS; i++){
        double x =skyPos[i][0];
        double z= skyPos[i][2];
        skyPos[i][1]=getGH(x, z)-2;
    }
}

void updateStreetLightPos(){
    for(int i=0;i<NUM_STREET_LIGHTS;i++){
        double x=stLtPos[i][0],z=stLtPos[i][2];
        stLtPos[i][1]=getGH(x,z);
    }
}
#define ATTACK_RANGE 5.0
int playerAttack = 0;

void mouseClick(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        playerAttack = 1;
    }
}

void checkZombieDamage() {
    if (playerAttack) {
        for (int i = 0; i < CZombies; i++) {
            double dist = sqrt(pow(cX - zX[i], 2) + pow(cZ - zZ[i], 2));
            if (dist < ATTACK_RANGE) {
                zX[i] = zY[i] = zZ[i] = -100;
            }
        }
        playerAttack = 0;
    }
}
void checkGameState(){
    double currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000;
    eTime = currentTime - sTime;

    if (playerHealth <= 0.0) {
        cGS = GAME_OVER;
    }

    if (eTime >= tLimit) {
        cGS = GAME_WIN;
    }
}
void drawGameOverScreen(){
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glColor3f(1.0, 0.0, 0.0);
    glRasterPos2i(winW/ 2- 50, winH / 2 + 20);
    Print("GAME OVER");
    glRasterPos2i(winW/2 - 70, winH / 2 - 20);
    Print("Press 'R' to Restart");

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}

void drawWinScreen(){
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glColor3f(0.0, 1.0, 0.0);
    glRasterPos2i(winW / 2 - 40, winH / 2 + 20);
    Print("YOU WIN!");
    glRasterPos2i(winW / 2 - 70, winH / 2 - 20);
    Print("Press 'R' to Restart");

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}


void drawHealthBar(float playerHealth){
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    // Draw health bar background
    glColor3f(0.2, 0.2, 0.2); 
    glBegin(GL_QUADS);
    glVertex2f(20, winH - 50);
    glVertex2f(220, winH - 50);
    glVertex2f(220, winH - 30);
    glVertex2f(20, winH - 30);
    glEnd();

    // Draw health bar foreground
    glColor3f(1.0 - playerHealth / 100.0, playerHealth / 100.0, 0.0); 
    glBegin(GL_QUADS);
    glVertex2f(20, winH - 50);
    glVertex2f(20 + (200 * (playerHealth / 100.0)), winH - 50);
    glVertex2f(20 + (200 * (playerHealth / 100.0)), winH - 30);
    glVertex2f(20, winH - 30);
    glEnd();

    // Restore state
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}


void drawGround(){
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex[8]);
    int gridSize=100;
    float maxH=2.0f;
    for(int x=-gridSize;x<gridSize;x++){
        glBegin(GL_TRIANGLE_STRIP);
        for(int z=-gridSize;z<=gridSize;z++){
            float xPos=x,zPos=z;
            float yPos1=sinf(xPos*0.2f)*cosf(zPos*0.2f)*maxH;
            float yPos2=sinf((xPos+1)*0.2f)*cosf(zPos*0.2f)*maxH;
            float nx=-cosf(xPos*0.2f)*cosf(zPos*0.2f);
            float nz=sinf(xPos*0.2f)*sinf(zPos*0.2f);
            float ny=1.0f;
            glNormal3f(nx,ny,nz);
            glTexCoord2f((x+gridSize)/(float)(gridSize*2),(z+gridSize)/(float)(gridSize*2));
            glVertex3f(xPos,yPos1,zPos);
            glNormal3f(nx,ny,nz);
            glTexCoord2f((x+1+gridSize)/(float)(gridSize*2),(z+gridSize)/(float)(gridSize*2));
            glVertex3f(xPos+1,yPos2,zPos);
        }
        glEnd();
    }
    glDisable(GL_TEXTURE_2D);
}

void drawBarricade(double x,double y,double z,double s,double rot){
    glPushMatrix();
    glTranslated(x,y,z);
    glRotated(rot,0,1,0);
    glScaled(s,s,s);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex[13]);
    glBegin(GL_QUADS);
    // First plank
    glNormal3f(0,0,1);
    glTexCoord2f(0,0); glVertex3f(-0.5,0,0.05);
    glTexCoord2f(1,0); glVertex3f(0.5,0,0.05);
    glTexCoord2f(1,1); glVertex3f(0.5,1.0,0.05);
    glTexCoord2f(0,1); glVertex3f(-0.5,1.0,0.05);
    // Second plank
    glNormal3f(0,0,1);
    glTexCoord2f(0,0); glVertex3f(-0.5,0,-0.05);
    glTexCoord2f(1,0); glVertex3f(0.5,0,-0.05);
    glTexCoord2f(1,1); glVertex3f(0.5,1.0,-0.05);
    glTexCoord2f(0,1); glVertex3f(-0.5,1.0,-0.05);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

void drawRock(double x,double y,double z,double s){
    glPushMatrix();
    glTranslated(x,y,z);
    glScaled(s,s,s);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex[14]);
    glPushMatrix();
    glScaled(1.0,0.8,1.2);
    drawSphere(0,0.5,0,0.5);
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

// Function to draw a house
void drawHouse(double x,double y,double z,double sz){
    glPushMatrix();
    glTranslated(x,y,z);
    glScaled(sz,sz,sz);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex[0]);
    glBegin(GL_QUADS);
    // Front face
    glNormal3f(0,0,1);
    glTexCoord2f(0,0); glVertex3f(-1,0,1);
    glTexCoord2f(1,0); glVertex3f(1,0,1);
    glTexCoord2f(1,1); glVertex3f(1,1,1);
    glTexCoord2f(0,1); glVertex3f(-1,1,1);
    // Back face
    glNormal3f(0,0,-1);
    glTexCoord2f(0,0); glVertex3f(1,0,-1);
    glTexCoord2f(1,0); glVertex3f(-1,0,-1);
    glTexCoord2f(1,1); glVertex3f(-1,1,-1);
    glTexCoord2f(0,1); glVertex3f(1,1,-1);
    // Right face
    glNormal3f(1,0,0);
    glTexCoord2f(0,0); glVertex3f(1,0,1);
    glTexCoord2f(1,0); glVertex3f(1,0,-1);
    glTexCoord2f(1,1); glVertex3f(1,1,-1);
    glTexCoord2f(0,1); glVertex3f(1,1,1);
    // Left face
    glNormal3f(-1,0,0);
    glTexCoord2f(0,0); glVertex3f(-1,0,1);
    glTexCoord2f(1,0); glVertex3f(-1,0,-1);
    glTexCoord2f(1,1); glVertex3f(-1,1,-1);
    glTexCoord2f(0,1); glVertex3f(-1,1,1);
    // Top face
    glNormal3f(0,1,0);
    glTexCoord2f(0,0); glVertex3f(-1,1,-1);
    glTexCoord2f(1,0); glVertex3f(1,1,-1);
    glTexCoord2f(1,1); glVertex3f(1,1,1);
    glTexCoord2f(0,1); glVertex3f(-1,1,1);
    // Bottom face
    glNormal3f(0,-1,0);
    glTexCoord2f(0,0); glVertex3f(-1,0,1);
    glTexCoord2f(1,0); glVertex3f(1,0,1);
    glTexCoord2f(1,1); glVertex3f(1,0,-1);
    glTexCoord2f(0,1); glVertex3f(-1,0,-1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D,tex[1]);
    // Draw the roof
    glBegin(GL_TRIANGLES);
    // Front roof
    glNormal3f(0,0.7071,0.7071);
    glTexCoord2f(0.5,1); glVertex3f(0,1.5,0);
    glTexCoord2f(0,0); glVertex3f(-1,1,1);
    glTexCoord2f(1,0); glVertex3f(1,1,1);
    // Back roof
    glNormal3f(0,0.7071,-0.7071);
    glTexCoord2f(0.5,1); glVertex3f(0,1.5,0);
    glTexCoord2f(1,0); glVertex3f(1,1,-1);
    glTexCoord2f(0,0); glVertex3f(-1,1,-1);
    // Left roof
    glNormal3f(-0.7071,0.7071,0);
    glTexCoord2f(0.5,1); glVertex3f(0,1.5,0);
    glTexCoord2f(1,0); glVertex3f(-1,1,1);
    glTexCoord2f(0,0); glVertex3f(-1,1,-1);
    // Right roof
    glNormal3f(0.7071,0.7071,0);
    glTexCoord2f(0.5,1); glVertex3f(0,1.5,0);
    glTexCoord2f(0,0); glVertex3f(1,1,-1);
    glTexCoord2f(1,0); glVertex3f(1,1,1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

void drawStreetLight(double x,double y,double z,double s){
    glPushMatrix();
    glTranslated(x,y,z);
    glScaled(s,s,s);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex[15]); // Metal texture
    // Draw the pole
    glColor3f(1.0f, 1.0f,1.0f);
    glPushMatrix();
    cylinder(0,0,0, 0.05,3.0,16);
    glPopMatrix();
    // Draw the lamp arm
    glPushMatrix();
    glTranslated(0, 3.0, 0);
    glRotated(-45,0,0, 1);
    cylinder(0,0, 0,0.03 ,1.0,16);
    glPopMatrix();
    // Lamp housing
    float emission[]={0.8f, 0.8f, 0.6f,1.0f};
    glMaterialfv(GL_FRONT,GL_EMISSION,emission);
    glPushMatrix();
    glTranslated(0.7,3.7,0);
    drawSphere(0,0,0,0.15);
    glPopMatrix();
    float noEmission[]={0.0f,0.0f,0.0f,1.0f};
    glMaterialfv(GL_FRONT,GL_EMISSION,noEmission);
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}
void drawZombie(double x, double y, double z, double s){
    double dx = cX - x;
    double dz = cZ - z;
    double angle = atan2(dz, dx) * 180 / M_PI; 

    glPushMatrix();
    glTranslated(x, y, z);           
    glRotated(-angle, 0, 1, 0);      
    glRotated(-90, 0, 1, 0);        
    glScaled(s, s, s);       

    glEnable(GL_TEXTURE_2D);

    // Draw head
    glBindTexture(GL_TEXTURE_2D, tex[9]); // Skin texture
    glPushMatrix();
    drawSphere(0, 1.8, 0, 0.3); // Head
    glPopMatrix();

    // Draw torso
    glBindTexture(GL_TEXTURE_2D, tex[10]); // Clothes texture
    glPushMatrix();
    glTranslated(0, 1.2, 0);
    glScaled(0.6, 1.0, 0.3); 
    drawSphere(0, 0, 0, 0.5);
    glPopMatrix();

    // Arms
    // Right Arm
    glPushMatrix();
    glTranslated(0.3, 1.5, 0); // Attach to shoulder
    glBindTexture(GL_TEXTURE_2D, tex[10]); 
    drawSphere(0, 0, 0, 0.1);  // Shoulder joint
    glPushMatrix();
    glRotated(30 * Sin(asa), 1, 0, 0); // Swing movement for walking
    cylinder(0, 0, 0, 0.1, -0.4, 20); // Upper arm
    glTranslated(0, -0.4, 0); // Move to elbow position
    glBindTexture(GL_TEXTURE_2D, tex[9]); 
    drawSphere(0, 0, 0, 0.1); // Elbow joint
    glRotated(30 * Sin(asa + 45), 1, 0, 0); // Bend at elbow
    cylinder(0, 0, 0, 0.1, -0.4, 20); // Lower arm
    glPopMatrix();
    glPopMatrix();

    // Left Arm
    glPushMatrix();
    glTranslated(-0.3, 1.5, 0); // Attach to shoulder
    glBindTexture(GL_TEXTURE_2D, tex[10]); 
    drawSphere(0, 0, 0, 0.1);  // Shoulder joint
    glPushMatrix();
    glRotated(-30 * Sin(asa), 1, 0, 0); // Opposite swing movement
    cylinder(0, 0, 0, 0.1, -0.4, 20); // Upper arm
    glTranslated(0, -0.4, 0); // Move to elbow position
    glBindTexture(GL_TEXTURE_2D, tex[9]); 
    drawSphere(0, 0, 0, 0.1); // Elbow joint
    glRotated(-30 * Sin(asa + 45), 1, 0, 0); // Bend at elbow
    cylinder(0, 0, 0, 0.1, -0.4, 20); // Lower arm
    glPopMatrix();
    glPopMatrix();

    // Legs
    // Right Leg
    glPushMatrix();
    glTranslated(0.15, 0.8, 0); // Attach to hip
    glBindTexture(GL_TEXTURE_2D, tex[10]); 
    drawSphere(0, 0, 0, 0.15);  // Hip joint
    glPushMatrix();
    glRotated(20 * Sin(asa), 1, 0, 0); // Swing movement for walking
    cylinder(0, 0, 0, 0.15, -0.5, 20); // Upper leg
    glTranslated(0, -0.5, 0); // Move to knee position
    glBindTexture(GL_TEXTURE_2D, tex[10]); 
    drawSphere(0, 0, 0, 0.14); // Knee joint
    cylinder(0, 0, 0, 0.14, -0.5, 20); // Lower leg
    glPopMatrix();
    glPopMatrix();

    // Left Leg
    glPushMatrix();
    glTranslated(-0.15, 0.8, 0); // Attach to hip
    glBindTexture(GL_TEXTURE_2D, tex[10]); 
    drawSphere(0, 0, 0, 0.15);  // Hip joint
    glPushMatrix();
    glRotated(-20 * Sin(asa), 1, 0, 0); // Opposite swing
    cylinder(0, 0, 0, 0.15, -0.5, 20); // Upper leg
    glTranslated(0, -0.5, 0); // Move to knee position
    glBindTexture(GL_TEXTURE_2D, tex[10]); 
    drawSphere(0, 0, 0, 0.14); // Knee joint
    cylinder(0, 0, 0, 0.14, -0.5, 20); // Lower leg
    glPopMatrix();
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

void updateZombiePos(){
    for(int i= 0; i<NUM_ZOMBIES; i++){
        double x=zX[i], z=zZ[i];
        double y=getGH(x,z);
        zY[i] = y;
    }
}

void drawHuman(double x, double y, double z, double s){
    glPushMatrix();
    glTranslated(x, y, z);
    glScaled(s, s, s);
    glEnable(GL_TEXTURE_2D);

    // Head
    glBindTexture(GL_TEXTURE_2D, tex[2]); // Skin texture
    glPushMatrix();
    drawSphere(0, 1.8, 0, 0.3);
    glPopMatrix();

    // Torso
    glBindTexture(GL_TEXTURE_2D, tex[3]); // Clothes texture
    glPushMatrix();
    cylinder(0, 1.0, 0, 0.3, 0.8, 20); 
    glPopMatrix();

    // Arms
    glBindTexture(GL_TEXTURE_2D, tex[2]); // Skin texture

    // Right Arm
    glPushMatrix();
    glTranslated(0.4, 1.7, 0); // Shoulder position
    // Upper Arm
    glPushMatrix();
    cylinder(0, 0, 0, 0.1, -0.4, 20); // From shoulder to elbow
    glPopMatrix();
    // Elbow Joint
    glPushMatrix();
    glTranslated(0, -0.4, 0); // Move to elbow position
    drawSphere(0, 0, 0, 0.11); // Elbow joint
    // Lower Arm
    cylinder(0, 0, 0, 0.09, -0.4, 20); // From elbow to hand
    glPopMatrix();
    glPopMatrix();

    // Left Arm
    glPushMatrix();
    glTranslated(-0.4, 1.7, 0); // Shoulder position
    // Upper Arm
    glPushMatrix();
    cylinder(0, 0, 0, 0.1, -0.4, 20); // From shoulder to elbow
    glPopMatrix();
    // Elbow Joint
    glPushMatrix();
    glTranslated(0, -0.4, 0); // Move to elbow position
    drawSphere(0, 0, 0, 0.11); // Elbow joint
    // Lower Arm
    cylinder(0, 0, 0, 0.09, -0.4, 20); // From elbow to hand
    glPopMatrix();
    glPopMatrix();

    // Legs
    glBindTexture(GL_TEXTURE_2D, tex[3]); // Clothes texture

    // Right Leg
    glPushMatrix();
    glTranslated(0.15, 1.0, 0); // Hip position
    // Upper Leg
    glPushMatrix();
    cylinder(0, 0, 0, 0.15, -0.5, 20); // From hip to knee
    glPopMatrix();
    // Knee Joint
    glPushMatrix();
    glTranslated(0, -0.5, 0); // Move to knee position
    drawSphere(0, 0, 0, 0.16); // Knee joint
    // Lower Leg
    cylinder(0, 0, 0, 0.14, -0.5, 20); // From knee to foot
    glPopMatrix();
    glPopMatrix();

    // Left Leg
    glPushMatrix();
    glTranslated(-0.15, 1.0, 0); // Hip position
    // Upper Leg
    glPushMatrix();
    cylinder(0, 0, 0, 0.15, -0.5, 20); // From hip to knee
    glPopMatrix();
    // Knee Joint
    glPushMatrix();
    glTranslated(0, -0.5, 0); // Move to knee position
    drawSphere(0, 0, 0, 0.16); // Knee joint
    // Lower Leg
    cylinder(0, 0, 0, 0.14, -0.5, 20); // From knee to foot
    glPopMatrix();
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}



void drawMoon(){
    double mX=m_r*Cos(m_ang);
    double mY=m_r *Sin(m_ang);
    double mZ=0.0;
    if(m_ang >=0&& m_ang <=180){
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D,tex[5]);
        glPushMatrix();
        drawSphere(mX,mY,mZ,2.0);
        glPopMatrix();
        glDisable(GL_TEXTURE_2D);
    }
}



void drawStar(double x,double y,double z,double sz){
    glPushMatrix();
    glTranslated(x,y,z);
    glRotated(sp_ang,0,1,0);
    glScaled(sz,sz,sz);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex[4]);
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0,0,1);
    glTexCoord2f(0.5,0.5); glVertex3f(0,0,0);
    for(int i=0;i<=10;i++){
        double angle=i*2*M_PI/10;
        double radius=(i%2==0)?1.0:0.5;
        double tx=0.5+radius*0.5*cos(angle);
        double ty=0.5+radius*0.5*sin(angle);
        glTexCoord2f(tx,ty);
        glVertex3f(radius*cos(angle),radius*sin(angle),0);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

void drawStarfield() {
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glColor3f(1.0, 1.0, 1.0); 
    glPointSize(1.0);

    glBegin(GL_POINTS);
    srand(0); 
    for (int i = 0; i<1000; i++) {
        double theta = (rand()% 360)*M_PI / 180.0;
        double phi = ((rand() %180)-90) * M_PI / 180.0;
        double radius = 200.0; 

        double x =radius*cos(phi)*cos(theta);
        double y=radius*sin(phi);
        double z=radius*cos(phi)*sin(theta);

        glVertex3d(x, y, z);
    }
    glEnd();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    glPopMatrix();
}

void drawSkyscraper(double x, double y, double z, double width, double height, double depth){
    glPushMatrix();
    glTranslated(x, y, z);
    glScaled(width, height, depth);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex[16]); 

    glBegin(GL_QUADS);
    // Front face
    glNormal3f(0, 0, 1);
    glTexCoord2f(0, 0); glVertex3f(-0.5, 0, 0.5);
    glTexCoord2f(1, 0); glVertex3f(0.5, 0, 0.5);
    glTexCoord2f(1, 1); glVertex3f(0.5, 1, 0.5);
    glTexCoord2f(0, 1); glVertex3f(-0.5, 1, 0.5);

    // Back face
    glNormal3f(0, 0, -1);
    glTexCoord2f(1, 0); glVertex3f(-0.5, 0, -0.5);
    glTexCoord2f(0, 0); glVertex3f(0.5, 0, -0.5);
    glTexCoord2f(0, 1); glVertex3f(0.5, 1, -0.5);
    glTexCoord2f(1, 1); glVertex3f(-0.5, 1, -0.5);

    // Left face
    glNormal3f(-1, 0, 0);
    glTexCoord2f(1, 0); glVertex3f(-0.5, 0, -0.5);
    glTexCoord2f(0, 0); glVertex3f(-0.5, 0, 0.5);
    glTexCoord2f(0, 1); glVertex3f(-0.5, 1, 0.5);
    glTexCoord2f(1, 1); glVertex3f(-0.5, 1, -0.5);

    // Right face
    glNormal3f(1, 0, 0);
    glTexCoord2f(0, 0); glVertex3f(0.5, 0, -0.5);
    glTexCoord2f(1, 0); glVertex3f(0.5, 0, 0.5);
    glTexCoord2f(1, 1); glVertex3f(0.5, 1, 0.5);
    glTexCoord2f(0, 1); glVertex3f(0.5, 1, -0.5);

    // Top face
    glNormal3f(0, 1, 0);
    glTexCoord2f(0, 1); glVertex3f(-0.5, 1, -0.5);
    glTexCoord2f(0, 0); glVertex3f(0.5, 1, -0.5);
    glTexCoord2f(1, 0); glVertex3f(0.5, 1, 0.5);
    glTexCoord2f(1, 1); glVertex3f(-0.5, 1, 0.5);

    // Bottom face 
    glNormal3f(0, -1, 0);
    glTexCoord2f(0, 0); glVertex3f(-0.5, 0, 0.5);
    glTexCoord2f(1, 0); glVertex3f(0.5, 0, 0.5);
    glTexCoord2f(1, 1); glVertex3f(0.5, 0, -0.5);
    glTexCoord2f(0, 1); glVertex3f(-0.5, 0, -0.5);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

void Projection(){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if(psp==0){
        glOrtho(-a*d,a*d,-d,d,-d*2,200);
    } else {
        gluPerspective(fov,a,0.1, 200);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
void spawnZombie(){
    if (CZombies< NUM_ZOMBIES){
        double newX = (rand() % 100)-50;
        double newZ = (rand() %100)- 50;
        if (sqrt(pow(newX- cX, 2)+ pow(newZ - cZ, 2))> 10.0) {
            zX[CZombies]=newX;
            zY[CZombies]= getGH(newX, newZ);
            zZ[CZombies]= newZ;
            CZombies++;
        }
    }
}

void setupFirstPersonView(){
    double dirX=Cos(cYaw)*Cos(cPitch);
    double dirY=Sin(cPitch);
    double dirZ=Sin(cYaw)*Cos(cPitch);
    gluLookAt(cX,cY,cZ,cX+dirX,cY+dirY,cZ+dirZ,0.0,1.0,0.0);
}
void setLighting(){
    float Ambient[]={0.01*amb,0.01*amb,0.02*amb,1.0};
    float Diffuse[]={0.01*diff,0.01*diff,0.01*diff,1.0};
    float Specular[]={0.01*spec,0.01*spec,0.01*spec,1.0};
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0,GL_AMBIENT,Ambient);
    glLightfv(GL_LIGHT0,GL_DIFFUSE,Diffuse);
    glLightfv(GL_LIGHT0,GL_SPECULAR,Specular);
    glLightf(GL_LIGHT0,GL_POSITION,5*Cos(l_ang));
    glLightf(GL_LIGHT0,GL_POSITION+1,ylt);
    glLightf(GL_LIGHT0,GL_POSITION+2,5*Sin(l_ang));
    glLightf(GL_LIGHT0,GL_POSITION+3,1.0);
    
    // Spotlight for first-person view
    glEnable(GL_LIGHT1);
    glLightf(GL_LIGHT1,GL_POSITION,cX);
    glLightf(GL_LIGHT1,GL_POSITION+1,cY);
    glLightf(GL_LIGHT1,GL_POSITION+2,cZ);
    glLightf(GL_LIGHT1,GL_POSITION+3,1.0);
    glLightf(GL_LIGHT1,GL_SPOT_DIRECTION,Cos(cYaw)*Cos(cPitch));
    glLightf(GL_LIGHT1,GL_SPOT_DIRECTION+1,Sin(cPitch));
    glLightf(GL_LIGHT1,GL_SPOT_DIRECTION+2,Sin(cYaw)*Cos(cPitch));
    glLightf(GL_LIGHT1,GL_SPOT_CUTOFF,30.0);
    glLightf(GL_LIGHT1,GL_SPOT_EXPONENT,2.0);
    glLightfv(GL_LIGHT1,GL_AMBIENT,(float[]){0,0,0,1});
    glLightfv(GL_LIGHT1,GL_DIFFUSE,(float[]){0.8,0.8,0.7,1});
    glLightfv(GL_LIGHT1,GL_SPECULAR,(float[]){0.9,0.9,0.8,1});

    // Street lights
    for(int i=0;i<NUM_STREET_LIGHTS;i++){
        GLenum lightID=GL_LIGHT2+i;
        if(lightID>GL_LIGHT7) break;
        glEnable(lightID);
        glLightf(lightID,GL_POSITION,stLtPos[i][0]+0.7);
        glLightf(lightID,GL_POSITION+1,stLtPos[i][1]+3.7);
        glLightf(lightID,GL_POSITION+2,stLtPos[i][2]);
        glLightf(lightID,GL_POSITION+3,1.0);
        glLightfv(lightID,GL_AMBIENT,(float[]){0,0,0,1});
        glLightfv(lightID,GL_DIFFUSE,(float[]){0.8,0.8,0.6,1});
        glLightfv(lightID,GL_SPECULAR,(float[]){0.9,0.9,0.7,1});
        glLightf(lightID,GL_CONSTANT_ATTENUATION,1.0);
        glLightf(lightID,GL_LINEAR_ATTENUATION,0.2);
        glLightf(lightID,GL_QUADRATIC_ATTENUATION,0.05);
        glLightf(lightID,GL_SPOT_DIRECTION,0);
        glLightf(lightID,GL_SPOT_DIRECTION+1,-1.0);
        glLightf(lightID,GL_SPOT_DIRECTION+2,0);
        glLightf(lightID,GL_SPOT_CUTOFF,45.0);
        glLightf(lightID,GL_SPOT_EXPONENT,10.0);
    }
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
}

// Main display function
void display(){
    glClearColor(0.0,0.0,0.05,1.0);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    if (cGS == RUNNING) {
    glLoadIdentity();
    if(psp==0){
        glTranslated(0.0,0.0,-d);
        glRotatef(p,1,0,0);
        glRotatef(t,0,1,0);
    } else if(psp==1){
        glTranslated(0.0,0.0,-d);
        glRotatef(p,1,0,0);
        glRotatef(t,0,1,0);
    } else if(psp==2){
        setupFirstPersonView();
    }
    drawStarfield();
    setLighting();
    // Draw houses
    for(int i=0;i<NUM_HOUSES;i++){
        double x=hPos[i][0],y=hPos[i][1],z=hPos[i][2];
        double sz=(i==0) ? 2.0:((i== 1)? 1.5:1.8);
        drawHouse(x,y,z,sz);
    }
    // Draw skyscrapers
    for (int i = 0; i < NUM_SKYSCRAPERS; i++) {
        double x= skyPos[i][0];
        double y= skyPos[i][1];
        double z= skyPos[i][2];
        double width=skyPos[i][3];
        double height=skyPos[i][4];
        double depth =skyPos[i][5];
        drawSkyscraper(x, y, z, width, height, depth);
    }
           
    // Draw airplane
    drawAirplane();
    // Draw ground
    drawGround();
    // Draw rocks
    for(int i=0;i<NUM_ROCKS;i++){
        double x=rPos[i][0],y=rPos[i][1],z=rPos[i][2];
        double s=(i==0)?0.7:((i==1)?0.5:0.9);
        drawRock(x,y,z,s);
    }
    // Draw street lights
    for(int i=0;i<NUM_STREET_LIGHTS;i++){
        drawStreetLight(stLtPos[i][0],stLtPos[i][1],stLtPos[i][2],1.0);
    }
    // Draw humans
    for(int i=0;i<NUM_HUMANS;i++){
        double x=huPos[i][0],y=huPos[i][1],z=huPos[i][2];
        double s=(i==0)?0.8:((i==1)?0.6:0.7);
        drawHuman(x,y,z,s);
    }
    // Draw trees
    drawTree(5,0,5,1.0);
    drawTree(-10,0,-8,1.2);
    drawTree(8,0,-5,0.9);
    drawTree(-6,0,7,1.1);
    drawTree(0,0,-10,1.3);
    for(int i = 0; i < NUM_ZOMBIES; i++){
        double x = zX[i];
        double y = zY[i];
        double z = zZ[i];
        drawZombie(x, y, z, 0.8);
    }

    // Draw stars
    drawStar(3,10,-4,0.3);
    drawStar(-2,9.5,5,0.25);
    drawStar(5,11,3,0.2);
    drawStar(-6,10.5,-5,0.3);
    drawStar(8,11.5,4,0.2);
    // Draw barricades
    for(int i=0;i<NUM_BARRICADES;i++){
        drawBarricade(barPos[i][0],barPos[i][1],barPos[i][2],1.0,1.0);
    }
    // Draw moon
    drawMoon();
    // Draw health bar
    drawHealthBar(playerHealth);
    // Draw axes
    if(axs){
        glDisable(GL_LIGHTING);
        glColor3f(1,1,1);
        glBegin(GL_LINES);
        glVertex3d(0.0,0.0,0.0);
        glVertex3d(5,0.0,0.0);
        glVertex3d(0.0,0.0,0.0);
        glVertex3d(-5,0.0,0.0);
        glVertex3d(0.0,0.0,0.0);
        glVertex3d(0.0,0.0,5);
        glVertex3d(0.0,0.0,0.0);
        glVertex3d(0.0,0.0,-5);
        glEnd();
        glRasterPos3d(5,0.0,0.0);
        Print("E");
        glRasterPos3d(-5,0.0,0.0);
        Print("W");
        glRasterPos3d(0.0,0.0,5);
        Print("S");
        glRasterPos3d(0.0,0.0,-5);
        Print("N");
        glEnable(GL_LIGHTING);
    }
    // Render text
    renderTextOnScreen();
    } else if (cGS == GAME_OVER) {
        drawGameOverScreen();
    } else if (cGS == GAME_WIN){
        drawWinScreen();
    }
    glutSwapBuffers();
}
void checkZombieAttack(){
    for (int i = 0; i < CZombies; i++){
        double dist = sqrt(pow(cX - zX[i], 2) + pow(cZ - zZ[i], 2));
        if (dist < ATTACK_RANGE / 2) {
            playerHealth -= 1;
            if (playerHealth <= 0) {
                printf("Game Over!\n");
                exit(0);
            }
        }
    }
}

// Check collision
int checkCollision(double x,double z){

    for (int i = 0; i < NUM_SKYSCRAPERS; i++){
        double dx = x - skyPos[i][0];
        double dz = z - skyPos[i][2];
        double dist = sqrt(dx * dx + dz * dz);
        if (dist < PLAYER_R + skyPos[i][3] * 0.5){
            return 1;
        }
    }
    // Check collision with houses
    for(int i=0;i<NUM_HOUSES;i++){
        double dx=x-hPos[i][0];
        double dz=z-hPos[i][2];
        double dist=sqrt(dx*dx+dz*dz);
        if(dist<PLAYER_R+HOUSE_R)
            return 1;
    }
    // Check collision with rocks
    for(int i=0;i<NUM_ROCKS;i++){
        double dx=x-rPos[i][0];
        double dz=z-rPos[i][2];
        double dist=sqrt(dx*dx+dz*dz);
        if(dist<PLAYER_R+ROCK_R)
            return 1;
    }
    // Check collision with trees
    double treePos[][3]={
        {5,0,5},{-10,0,-8},{8,0,-5},{-6,0,7},{0,0,-10}
    };
    int numTrees=sizeof(treePos)/sizeof(treePos[0]);
    for(int i=0;i<numTrees;i++){
        double dx=x-treePos[i][0];
        double dz=z-treePos[i][2];
        double dist=sqrt(dx*dx+dz*dz);
        if(dist<PLAYER_R+TREE_R)
            return 1;
    }
    return 0;
}


void updateZombies(){
    for (int i = 0; i < CZombies; i++){
        double dx = cX - zX[i];
        double dz = cZ - zZ[i];
        double dist = sqrt(dx * dx + dz * dz);

        if (dist > 0.1 && dist< 50.0){
            double dirX = dx /dist;
            double dirZ = dz/ dist;

            double newX= zX[i] + dirX * .15;
            double newZ = zZ[i] + dirZ * .15;

            if (!checkCollision(newX, newZ)) {
                zX[i] = newX;
                zZ[i] =newZ;
            } else {
                zX[i]+= (rand() % 3 -1) * 0.1; 
                zZ[i] += (rand() % 3 - 1) * 0.1;
            }

            zY[i] = getGH(zX[i], zZ[i]);
        }
    }
}

int zombieHitsPlayer(){
    for (int i = 0; i< CZombies; i++){
        double dx= cX- zX[i];
        double dz= cZ - zZ[i];
        double distance = sqrt(dx *dx + dz * dz);

        if (distance < 1.0){
            return 1;
        }
    }
    return 0;
}

void mouseMovement(int x, int y){
    static int lastX, lastY;
    if (lastX == 0 && lastY == 0){
        lastX= winW/ 2;
        lastY =winH / 2;
}
    int deltaX=x - lastX;
    int deltaY= y - lastY;
    cYaw += deltaX * 0.1; 
    cPitch -= deltaY * 0.1;
    if (cPitch> 89.0) cPitch = 89.0;
    if (cPitch <-89.0) cPitch = -89.0;

    glutWarpPointer(winW /2, winH/ 2);

    lastX= winW/ 2;
    lastY= winH/ 2;

    glutPostRedisplay();
}


void updateHealth(float damage) {
    playerHealth -= damage;
    if (playerHealth < 0) playerHealth = 0;
}

//Idle callback for animation
void idle(){
    double cTime=glutGet(GLUT_ELAPSED_TIME)/1000;
    checkZombieDamage();
    static int spawnTimer = 0;
    spawnTimer++;
    if (spawnTimer > 200){
        spawnZombie();
        spawnTimer = 0;
    }
    sp_ang+=0.2;
    if(sp_ang>360)
        sp_ang-=360;
    m_ang+=0.01;
    if(m_ang>360)
        m_ang-=360;
    ap_ang+=0.005;
    if(ap_ang>360)
        ap_ang-=360;
    asa += zspeed * 1000;
    if (asa > 360)
        asa -= 360;
    lsa += zspeed * 1000; 
    if (lsa > 360)
        lsa -= 360;
   
    if (cGS == RUNNING) {
        sp_ang += 0.2;
        if (sp_ang > 360) sp_ang -= 360;

        asa += 2.0;
        if (asa > 360) asa -= 360;

        updateZombies();

        if (zombieHitsPlayer() && (cTime - lHit > 2.0)){ 
            updateHealth(10.0);
            lHit=cTime;
        }

        checkGameState(); 
    }
    glutPostRedisplay();
}

//Handle special keys

void special(int key,int x,int y){
    if(l_move){
        // Light control mode
        if(key==GLUT_KEY_RIGHT)
            l_ang+=5;
        else if(key==GLUT_KEY_LEFT)
            l_ang-=5;
        else if(key==GLUT_KEY_UP)
            ylt+=0.2;
        else if(key==GLUT_KEY_DOWN)
            ylt-=0.2;
    } else {
        // Camera control mode
        if(psp==2){
            if(key==GLUT_KEY_UP) cPitch+=5;
            else if(key==GLUT_KEY_DOWN) cPitch-=5;
            else if(key==GLUT_KEY_LEFT) cYaw-=5;
            else if(key==GLUT_KEY_RIGHT) cYaw+=5;
        } else {
            if(key==GLUT_KEY_RIGHT) t+=5;
            else if(key==GLUT_KEY_LEFT) t-=5;
            else if(key==GLUT_KEY_UP) p+=5;
            else if(key==GLUT_KEY_DOWN) p-=5;
        }
        t%=360;
        p%=360;
        cYaw=fmod(cYaw,360.0);
        cPitch=fmod(cPitch,360.0);
    }
    glutPostRedisplay();
}
void restartGame() {
    // Reset everything
    cGS = RUNNING;
    playerHealth = 100.0;
    sTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    eTime = 0.0;

    cX =0.0;
    cY =1.8;
    cZ= 8.0;
    cYaw =0.0;
    cPitch = 0.0;
  
    CZombies = 2;
    zX[0]=5.0; zZ[0]=5.0; zY[0]=getGH(zX[0],zZ[0]);
    zX[1] =-5.0; zZ[1]=-5.0; zY[1]=getGH(zX[1], zZ[1]);


    for (int i=2; i<NUM_ZOMBIES; i++){
        zX[i]=zY[i]=zZ[i]=-100.0;
    }

    for (int i=0; i<NUM_BARRICADES; i++){
        barPos[i][0] = (rand() % 100)-50;
        barPos[i][2] = (rand() % 100)-50;
        barPos[i][1] = getGH(barPos[i][0], barPos[i][2]);
    }

  
    for (int i =0; i<NUM_HUMANS;i++){
        huPos[i][0]=(rand()%100)-50;
        huPos[i][2]=(rand()%100)-50;
        huPos[i][1]=getGH(huPos[i][0], huPos[i][2]);
    }


    for (int i = 0; i<NUM_ROCKS;i++) {
        rPos[i][0]=(rand() % 100)-50;
        rPos[i][2]=(rand() % 100)-50;
        rPos[i][1]=getGH(rPos[i][0],rPos[i][2]);
    }

  
    InitSkyPos();
    updateSkyscraperPos();

    ap_ang = 0.0;
    m_ang = 0.0;
    updateStreetLightPos();
    updateBarPos();
    updateHousePos();
    updateRockPos();
    updateHumanPos();
    updateZombiePos();

    Projection();

    glutPostRedisplay();
}

//Handle regular keys for movement and mode switches

void key(unsigned char ch,int x,int y){
    if(ch==27){
        exit(0);
    } else if (ch == 'r' || ch == 'R') { // Restart key
        restartGame();
    }else if(ch=='m'||ch=='M'){
        psp=(psp+1)%3;
        Projection();
    } else if(ch=='v'||ch=='V'){
        l_move=1-l_move;
    } else if(ch=='b'||ch=='B'){
        axs=1-axs;
    } else if(ch=='+'&&fov<179){
        fov++;
        Projection();
    } else if(ch=='-'&&fov>1){
        fov--;
        Projection();
    }
    double mv_spd=0.5;
    if(l_move){
        if(ch=='w'||ch=='W')
            ylt+=0.2;
        else if(ch=='s'||ch=='S')
            ylt-=0.2;
        else if(ch=='a'||ch=='A')
            l_ang+=5;
        else if(ch=='d'||ch=='D')
            l_ang-=5;
    } else {
        if(psp==2){
            double dirX=Cos(cYaw)*Cos(cPitch);
            double dirZ=Sin(cYaw)*Cos(cPitch);
            double deltaX=0.0,deltaZ=0.0;
            if(ch=='w'||ch=='W'){
                deltaX=mv_spd*dirX;
                deltaZ=mv_spd*dirZ;
            } else if(ch=='s'||ch=='S'){
                deltaX=-mv_spd*dirX;
                deltaZ=-mv_spd*dirZ;
            } else if(ch=='a'||ch=='A'){
                deltaX=mv_spd*Cos(cYaw-90);
                deltaZ=mv_spd*Sin(cYaw-90);
            } else if(ch=='d'||ch=='D'){
                deltaX=mv_spd*Cos(cYaw+90);
                deltaZ=mv_spd*Sin(cYaw+90);
            }
            double newX=cX+deltaX;
            double newZ=cZ+deltaZ;
            if(!checkCollision(newX,newZ)){
                cX=newX;
                cZ=newZ;
            }
            cY=getGH(cX,cZ)+1.8;
            if(cY<getGH(cX,cZ)+1.8){
                cY=getGH(cX,cZ)+1.8;
            }
        } else {
            if(ch=='w'||ch=='W')
                d-=0.5;
            else if(ch=='s'||ch=='S')
                d+=0.5;
            d=fmax(d,2.0);
            Projection();
        }
    }
    glutPostRedisplay();
}

// Window reshape callback
void reshape(int width,int height){
    winW=width;
    winH=height;
    a=(height>0)?(double)width/height:1;
    glViewport(0,0,width,height);
    Projection();
}

//Main function
int main(int argc,char* argv[]){
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE|GLUT_DEPTH);
    glutInitWindowSize(1200,1200);
    glutCreateWindow("3D Scene with Custom Projections and Controls");
#ifdef USEGLEW
    if(glewInit()!=GLEW_OK){
        fprintf(stderr,"Error initializing GLEW\n");
        return -1;
    }
#endif
    glEnable(GL_DEPTH_TEST);
    loadTextures();
    sTime = glutGet(GLUT_ELAPSED_TIME) / 1000;

    updateStreetLightPos();
    InitSkyPos();
    updateSkyscraperPos();
    updateBarPos();
    updateHousePos();
    updateRockPos();
    updateHumanPos();
    updateZombiePos();
    glutDisplayFunc(display);
    glutPassiveMotionFunc(mouseMovement);
    glutReshapeFunc(reshape);
    glutSpecialFunc(special);
    glutSetCursor(GLUT_CURSOR_NONE);
    glutKeyboardFunc(key);
    glutIdleFunc(idle);
    glutMainLoop();
    return 0;
}
