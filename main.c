#include <stdio.h>
#include <hdf5.h>
#include <mpi.h>
#include <stdlib.h>


// Boundary conditions
#define XPERIODIC
#undef YPERIODIC

// Domain definition
#define NX 32
#define NY 128
#define NP 9

// Time
#define NTIME   100000
#define NSTORE  1000
#define NLOG    10

// Array indexing
#define INDEX_2D(i,j)       ((NY_proc+2)*(i+1) + (j+1))
#define INDEX_3D(i,j,p)     ((NY_proc+2)*NP*(i+1) + NP*(j+1) + (p))

// Parameters
double tau = 1.0;
double F_p = 1e-6;

// Stencil
double cs2 = 1.0/3.0;
int cx[] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
int cy[] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
int p_bounceback[] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
double w[] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

hid_t open_output(int t, int n_output) {
    char filename[32];
    sprintf(filename, "data_%d.h5", n_output);
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);

    H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    hid_t scalar_space = H5Screate(H5S_SCALAR);

    hid_t dset_scalar = H5Dcreate2(file_id, "t", H5T_NATIVE_INT, scalar_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dset_scalar, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &t);

    H5Dclose(dset_scalar);
    H5Sclose(scalar_space);

    return file_id;
}

void output_data(double* field, char *fieldname, hid_t file_id, int i_start, int i_end, int j_start, int j_end) {

    int NX_proc = i_end-i_start;
    int NY_proc = j_end-j_start;

    hsize_t dims_file[2] = {NX, NY};
    hid_t filespace = H5Screate_simple(2, dims_file, NULL);

    hsize_t dims_proc[2] = {NX_proc+2, NY_proc+2};
    hid_t memspace  = H5Screate_simple(2, dims_proc, NULL);

    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    hid_t dset_id = H5Dcreate2(file_id, fieldname, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    H5Pclose(dcpl_id);
    H5Sclose(filespace);

    hsize_t start_proc[2] = {1, 1};
    hsize_t count[2] = {NX_proc, NY_proc};    
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start_proc, NULL, count, NULL);

    hsize_t start_file[2] = {i_start, j_start};
    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start_file, NULL, count, NULL);

    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl_id, field);

    H5Dclose(dset_id);
    H5Sclose(memspace);
    H5Pclose(dxpl_id);
    H5Sclose(filespace);
}

void close_output(hid_t file_id) {
    H5Fflush(file_id, H5F_SCOPE_GLOBAL);
    H5Fclose(file_id);
}

int mod(int x, int n) {
    if (x < 0) return x+n;
    else if (x > n-1) return x-n;
    return x;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int number_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    int dims[2] = {1, number_of_processes};

    int periods[2] = {0};
#ifdef XPERIODIC
    periods[0] = 1;
#endif
#ifdef YPERIODIC
    periods[1] = 1;
#endif
    int reorder = 1;
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm_cart);

    int process_rank;
    MPI_Comm_rank(comm_cart, &process_rank);

    int coords[2] = {0};
    MPI_Cart_get(comm_cart, 2, dims, periods, coords);

    int processor_neighbors_x[2];
    int processor_neighbors_y[2];
    MPI_Cart_shift(comm_cart, 0, 1, &processor_neighbors_x[0], &processor_neighbors_x[1]);
    MPI_Cart_shift(comm_cart, 1, 1, &processor_neighbors_y[0], &processor_neighbors_y[1]);

    int i_start = (float)(coords[0])/(float)(dims[0])*NX;
    int i_end = (float)(coords[0]+1)/(float)(dims[0])*NX;
    int j_start = (float)(coords[1])/(float)(dims[1])*NY;
    int j_end = (float)(coords[1]+1)/(float)(dims[1])*NY;

    int NX_proc = i_end-i_start;
    int NY_proc = j_end-j_start;

    double *rho, *u, *v;
    double *f1, *f2;

    double memsize_field = (NX_proc+2)*(NY_proc+2)*sizeof(double);
    rho = (double *) malloc(memsize_field);
    u = (double *) malloc(memsize_field);
    v = (double *) malloc(memsize_field);

    double *send_buffer_x = (double *) malloc(NY_proc*NP*sizeof(double));
    double *recv_buffer_x = (double *) malloc(NY_proc*NP*sizeof(double));
    double *send_buffer_y = (double *) malloc(NX_proc*NP*sizeof(double));
    double *recv_buffer_y = (double *) malloc(NX_proc*NP*sizeof(double));

    double *buffer_TL = (double *) malloc(NP*sizeof(double));
    double *buffer_TR = (double *) malloc(NP*sizeof(double));
    double *buffer_BL = (double *) malloc(NP*sizeof(double));
    double *buffer_BR = (double *) malloc(NP*sizeof(double));

    double memsize_dist = (NX_proc+2)*(NY_proc+2)*(NP)*sizeof(double);
    f1 = (double *) malloc(memsize_dist);
    f2 = (double *) malloc(memsize_dist);

    // Initialize fields
    for (int i = 0; i < NX_proc; i++) {
        for (int j = 0; j < NY_proc; j++) {
            rho[INDEX_2D(i,j)] = 1.0;
            u[INDEX_2D(i,j)] = 0.0;
            v[INDEX_2D(i,j)] = 0.0;
        }
    }

    // Initialize distributions
    double uhat, u2, uc;
    uhat = -0.5*F_p;
    u2 = uhat*uhat;
    for (int i = 0; i < NX_proc; i++) {
        for (int j = 0; j < NY_proc; j++) {
            for (int p = 0; p < NP; p++) {
                uc = uhat*cx[p];
                f1[INDEX_3D(i,j,p)] = w[p]*(1.0 + uc/cs2 + (uc*uc)/(2.0*cs2*cs2) - u2/(2.0*cs2));
            }
        }
    }

    int t = 0;
    int n_output = 0;
    int t_output = NSTORE;
    hid_t file_id;
    
    int t_log = 0;

    file_id = open_output(t, n_output);
    output_data(rho, "rho", file_id, i_start, i_end, j_start, j_end);
    output_data(u, "u", file_id, i_start, i_end, j_start, j_end);
    output_data(v, "v", file_id, i_start, i_end, j_start, j_end);
    close_output(file_id);
    n_output++;

    MPI_Status status_first;

    double rho_i, u_i, v_i, feq, S;
    int ic, jc;

    while (t < NTIME) {
        
        // Output
        if (t == t_output) {
            file_id = open_output(t, n_output);
            output_data(rho, "rho", file_id, i_start, i_end, j_start, j_end);
            output_data(u, "u", file_id, i_start, i_end, j_start, j_end);
            output_data(v, "v", file_id, i_start, i_end, j_start, j_end);
            close_output(file_id);
            n_output++;
            t_output += NSTORE;
        }

        // Collision
        for (int i = 0; i < NX_proc; i++) {
            for (int j = 0; j < NY_proc; j++) {
                rho_i = rho[INDEX_2D(i,j)];
                u_i = u[INDEX_2D(i,j)];
                v_i = v[INDEX_2D(i,j)];
                u2 = u_i*u_i + v_i*v_i;
                for (int p = 0; p < NP; p++) {
                    uc = u_i*(double)cx[p] + v_i*(double)cy[p];
                    feq = w[p]*rho_i*(1.0 + uc/cs2 + (uc*uc)/(2.0*cs2*cs2) - u2/(2.0*cs2));
                    S = (1.0 - 1.0/(2.0*tau))*w[p]*(((double)cx[p]-u_i)/cs2 + uc/(cs2*cs2)*(double)cx[p])*F_p;
                    f2[INDEX_3D(i,j,p)] = (1.0 - 1.0/tau)*f1[INDEX_3D(i,j,p)] + 1.0/tau*feq + S;
                }
            }
        }

        // LEFT
        if (processor_neighbors_x[0] != MPI_PROC_NULL) {
            for (int j = 0; j < NY_proc; j++) {
                for (int p = 0; p < NP; p++) {
                    send_buffer_x[NP*j + p] = f2[INDEX_3D(0,j,p)];   
                }
            }
        }
        MPI_Sendrecv(send_buffer_x, NY_proc*NP, MPI_DOUBLE, processor_neighbors_x[0], 1, recv_buffer_x, NY_proc*NP, MPI_DOUBLE, processor_neighbors_x[1], 1, comm_cart, &status_first);
        if (processor_neighbors_x[1] != MPI_PROC_NULL) {
            for (int j = 0; j < NY_proc; j++) {
                for (int p = 0; p < NP; p++) {
                    f2[INDEX_3D(NX_proc,j,p)] = recv_buffer_x[NP*j + p];   
                }
            }
        }

        // RIGHT
        if (processor_neighbors_x[1] != MPI_PROC_NULL) {
            for (int j = 0; j < NY_proc; j++) {
                for (int p = 0; p < NP; p++) {
                    send_buffer_x[NP*j + p] = f2[INDEX_3D(NX_proc-1,j,p)];   
                }
            }
        }
        MPI_Sendrecv(send_buffer_x, NY_proc*NP, MPI_DOUBLE, processor_neighbors_x[1], 2, recv_buffer_x, NY_proc*NP, MPI_DOUBLE, processor_neighbors_x[0], 2, comm_cart, &status_first);
        if (processor_neighbors_x[0] != MPI_PROC_NULL) {
            for (int j = 0; j < NY_proc; j++) {
                for (int p = 0; p < NP; p++) {
                    f2[INDEX_3D(-1,j,p)] = recv_buffer_x[NP*j + p];   
                }
            }
        }

        // BOTTOM
        if (processor_neighbors_y[0] != MPI_PROC_NULL) {
            for (int i = 0; i < NX_proc; i++) {
                for (int p = 0; p < NP; p++) {
                    send_buffer_y[NP*i + p] = f2[INDEX_3D(i,0,p)];   
                }
            }
        }
        MPI_Sendrecv(send_buffer_y, NX_proc*NP, MPI_DOUBLE, processor_neighbors_y[0], 3, recv_buffer_y, NX_proc*NP, MPI_DOUBLE, processor_neighbors_y[1], 3, comm_cart, &status_first);
        if (processor_neighbors_y[1] != MPI_PROC_NULL) {
            for (int i = 0; i < NX_proc; i++) {
                for (int p = 0; p < NP; p++) {
                    f2[INDEX_3D(i,NY_proc,p)] = recv_buffer_y[NP*i + p];   
                }
            }
        }

        // TOP
        if (processor_neighbors_y[1] != MPI_PROC_NULL) {
            for (int i = 0; i < NX_proc; i++) {
                for (int p = 0; p < NP; p++) {
                    send_buffer_y[NP*i + p] = f2[INDEX_3D(i,NY_proc-1,p)];   
                }
            }
        }
        MPI_Sendrecv(send_buffer_y, NX_proc*NP, MPI_DOUBLE, processor_neighbors_y[1], 4, recv_buffer_y, NX_proc*NP, MPI_DOUBLE, processor_neighbors_y[0], 4, comm_cart, &status_first);
        if (processor_neighbors_y[0] != MPI_PROC_NULL) {
            for (int i = 0; i < NX_proc; i++) {
                for (int p = 0; p < NP; p++) {
                    f2[INDEX_3D(i,-1,p)] = recv_buffer_y[NP*i + p];   
                }
            }
        }
        
        for (int i = 0; i < NX_proc; i++) {
            for (int j = 0; j < NY_proc; j++) {
                for (int p = 0; p < NP; p++) {
                    ic = i-cx[p];
                    jc = j-cy[p];

                    if (((jc < 0) && (j_start == 0)) ||
                        ((jc == NY_proc) && (j_end == NY))) {
                        f1[INDEX_3D(i,j,p)] = f2[INDEX_3D(i, j, p_bounceback[p])];
                    }
                    else {
                        f1[INDEX_3D(i,j,p)] = f2[INDEX_3D(ic, jc, p)];
                    }
                }
            }
        }

        for (int i = 0; i < NX_proc; i++) {
            for (int j = 0; j < NY_proc; j++) {
                rho_i = 0.0;
                u_i = 0.0;
                v_i = 0.0;
                for (int p = 0; p < NP; p++) {
                    rho_i += f1[INDEX_3D(i,j,p)];
                    u_i += f1[INDEX_3D(i,j,p)]*(double)cx[p];
                    v_i += f1[INDEX_3D(i,j,p)]*(double)cy[p];
                }
                rho[INDEX_2D(i,j)] = rho_i;
                u[INDEX_2D(i,j)] = u_i/rho_i + F_p/(2.0*rho_i);
                v[INDEX_2D(i,j)] = v_i/rho_i;
            }
        }


        if (t == t_log) {
            if (process_rank == 0) {
                printf("\rProgress: %.2f", ((double)(t+1))/((double)NTIME)*100.0);
                fflush(stdout);
            }
            t_log += NLOG;
        }

        t++;
    }

    file_id = open_output(t, n_output);
    output_data(rho, "rho", file_id, i_start, i_end, j_start, j_end);
    output_data(u, "u", file_id, i_start, i_end, j_start, j_end);
    output_data(v, "v", file_id, i_start, i_end, j_start, j_end);
    close_output(file_id);

    if (process_rank == 0) {
        printf("\nDone!\n");
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}