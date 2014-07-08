/* Author: Christopher Mitchell <chrism@lclark.edu>
 * Date: 2011-08-10
 *
 * Compile with `gcc -lX11 gol.c`.
 */

#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <X11/Xlib.h> // for the graphics

#define WIDTH 800
#define HEIGHT 600

// The two boards 
int current[WIDTH * HEIGHT];
int next[WIDTH * HEIGHT];

const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
                           {-1, 0},       {1, 0},
                           {-1,-1},{0,-1},{1,-1}};


void fill_board(int *board) {
    int i;
    for (i=0; i<WIDTH*HEIGHT; i++)
        board[i] = rand() % 2;
}

void step() {
    // coordinates of the cell we're currently evaluating
    int x, y;
    // offset index, neighbor coordinates, alive neighbor count
    int i, nx, ny, num_neighbors;

    // write the next board state
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            // count this cell's alive neighbors
            num_neighbors = 0;
            for (i=0; i<8; i++) {
                // To make the board torroidal, we use modular arithmetic to
                // wrap neighbor coordinates around to the other side of the
                // board if they fall off.
                nx = (x + offsets[i][0] + WIDTH) % WIDTH;
                ny = (y + offsets[i][1] + HEIGHT) % HEIGHT;
                if (current[ny * WIDTH + nx]) {
                    num_neighbors++;
                }
            }

            // apply the Game of Life rules to this cell
            next[y * WIDTH + x] = 0;
            if ((current[y * WIDTH + x] && num_neighbors==2) ||
                    num_neighbors==3) {
                next[y * WIDTH + x] = 1;
            }
        }
    }
}

void animate() {
    Display* display;
    display = XOpenDisplay(NULL);
    if (display == NULL) {
        fprintf(stderr, "Could not open an X display.\n");
        exit(-1);
    }
    int screen_num = DefaultScreen(display);

    int black = BlackPixel(display, screen_num);
    int white = WhitePixel(display, screen_num);

    Window win = XCreateSimpleWindow(display,
            RootWindow(display, screen_num),
            0, 0,
            WIDTH, HEIGHT,
            0,
            black, white);
    XStoreName(display, win, "The Game of Life");

    XSelectInput(display, win, StructureNotifyMask);
    XMapWindow(display, win);
    while (1) {
        XEvent e;
        XNextEvent(display, &e);
        if (e.type == MapNotify)
            break;
    }

    GC gc = XCreateGC(display, win, 0, NULL);

    int x, y, n;
    XPoint points[WIDTH * HEIGHT];
    while (1) {
        XClearWindow(display, win);
        n = 0;
        for (y=0; y<HEIGHT; y++) {
            for (x=0; x<WIDTH; x++) {
                if (current[y * WIDTH + x]) {
                    points[n].x = x;
                    points[n].y = y;
                    n++;
                }
            }
        }
        XDrawPoints(display, win, gc, points, n, CoordModeOrigin);
        XFlush(display);

        step();
        // Copy the next state, that step() just wrote into, to current state
        memcpy(current, next, sizeof(int) * WIDTH * HEIGHT);
    }
}


int main(void) {
    // Initialize the global "current".
    fill_board(current);
    animate();

    return 0;
}
