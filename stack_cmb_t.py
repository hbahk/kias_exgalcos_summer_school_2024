import numpy as np
from tqdm import tqdm, trange
import multiprocessing as mp
import camb

# Assuming hp and its functions are available as imported
import healpy as hp

def get_alms(powers, label):
    # get the power spectrum
    aCls = powers[label]

    lmax = aCls.shape[0]-1
    aL = np.arange(lmax+1)

    aClTT = aCls[:,0]/aL/(aL+1)*2*np.pi
    aClTT[0]=0;aClTT[1]=0
    aClEE = aCls[:,1]/aL/(aL+1)*2*np.pi
    aClEE[0]=0;aClEE[1]=0
    aClBB = aCls[:,2]/aL/(aL+1)*2*np.pi
    aClBB[0]=0;aClBB[1]=0
    aClTE = aCls[:,3]/aL/(aL+1)*2*np.pi
    aClTE[0]=0;aClTE[1]=0

    alms_T, alms_E, alms_B = hp.synalm([aClTT,aClEE,aClBB,aClTE],lmax=2000,new=True,)
    
    return alms_T, alms_E, alms_B

def get_tqu_maps(alms_T, alms_E, alms_B):
    # generate the map
    TQU_maps = hp.alm2map([alms_T,alms_E,alms_B],lmax=2000,nside=512)
    # T,Q,U map
    T_map = TQU_maps[0]
    Q_map = TQU_maps[1]
    U_map = TQU_maps[2]
    
    return T_map, Q_map, U_map

def get_smoothed_maps(T_map, Q_map, U_map):
    tmap_smth = hp.smoothing(T_map, fwhm=0.5*np.pi/180)
    qmap_smth = hp.smoothing(Q_map, fwhm=0.5*np.pi/180)
    umap_smth = hp.smoothing(U_map, fwhm=0.5*np.pi/180)
    
    return tmap_smth, qmap_smth, umap_smth

def get_hot_coldspots(map, sort=True):
    spotmap, coldspots, hotspots = hp.hotspots(map)
    if sort:
        hsort = np.argsort(map[hotspots])[::-1]
        csort = np.argsort(map[coldspots])
        
        return coldspots[csort], hotspots[hsort]
    else:
        return coldspots, hotspots
    
import numpy as np
import healpy as hp
from tqdm import tqdm
from multiprocessing import Process, Manager

# Function to process each chunk of data
def process_chunk(tsm, lons, lats, xsize, shared_maps, start_index, end_index):
    local_maps = []
    for i in trange(start_index, end_index):
        local_map = hp.visufunc.gnomview(tsm, rot=(lons[i], lats[i], 0), xsize=xsize, return_projected_map=True, no_plot=True)
        local_maps.append(local_map)
    shared_maps.extend(local_maps)  # Append results to the shared list


def main():
    nside = 512
    xsize = 200
    num_processes = 32

    # Dummy data for demonstration purposes
    # Load actual data as needed
    # Use the provided planck2018 initial file
    planck2018pars = camb.read_ini("planck_2018.ini")
    planck2018 = camb.get_results(planck2018pars)
    # get the power spectrum
    powers = planck2018.get_cmb_power_spectra(planck2018pars,CMB_unit='muK')

    tm1, qm1, um1 = get_tqu_maps(*get_alms(powers, 'unlensed_scalar'))

    tsm1, qsm1, usm1 = get_smoothed_maps(tm1, qm1, um1)

    c1, h1 = get_hot_coldspots(tsm1)
    
    hlons1, hlats1 = hp.pix2ang(nside, ipix=h1, lonlat=True)
    clons1, clats1 = hp.pix2ang(nside, ipix=c1, lonlat=True)

    manager = Manager()
    hmaps = manager.list()
    cmaps = manager.list()

    # Create and start processes
    processes = []
    chunk_size = len(hlons1) // num_processes
    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = start_index + chunk_size if i < num_processes - 1 else len(hlons1) -1
        p = Process(target=process_chunk, args=(tsm1, hlons1, hlats1, xsize, hmaps, start_index, end_index))
        p.start()
        processes.append(p)
        p = Process(target=process_chunk, args=(tsm1, clons1, clats1, xsize, cmaps, start_index, end_index))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Convert shared list to array and save
    hmap = np.array(list(hmaps))
    cmap = np.array(list(cmaps))
    np.save("hotmap.npy", hmap)
    np.save("coldmap.npy", cmap)
    

if __name__ == "__main__":
    main()
