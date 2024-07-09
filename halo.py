import numpy as np

def read_halocat(halocatpath, amin=1):
    
    # define the Halo structure assuming you know the fields and their types
    # XYZDBL is defined and coordinates are stored as doubles
    dtype_halo = np.dtype(
        [
            ("np", np.uint64),
            ("x", np.float64),  # double precision float
            ("y", np.float64),
            ("z", np.float64),
            ("vx", np.float32),
            ("vy", np.float32),
            ("vz", np.float32),
            ("pad1", "byte", 4),
        ]
    )

    # open the halo catalog file
    with open(halocatpath, "rb") as hid:
        # read the single value fields
        size, hubble, npower = np.fromfile(hid, dtype=np.float32, count=3)
        omegp, omegpb, omegplam = np.fromfile(hid, dtype=np.float32, count=3)
        wlam0, wlam1, bias = np.fromfile(hid, dtype=np.float32, count=3)
        nx, nspace = np.fromfile(hid, dtype=np.int32, count=2)
        amax, astep, anow = np.fromfile(hid, dtype=np.float32, count=3)
        istep = (anow-amin)/astep + 1

        # read one halo structure
        halo = np.fromfile(hid, dtype=dtype_halo, count=-1)
        snapshot_props = {'size':size, 'hubble':hubble, 'npower':npower, 'omegp':omegp,
                 'omegpb':omegpb, 'omegplam':omegplam, 'wlam0':wlam0,
                 'wlam1':wlam1, 'bias':bias, 'nx':nx, 'nspace':nspace,
                 'amax':amax, 'astep':astep, 'anow':anow, 'istep':istep}
    
    return halo, snapshot_props


def read_haloqcat(haloqcatpath):
    dtype_halo_prop = np.dtype(
        [
            ("hid", np.int64),
            ("mbp", np.int64),
            ("mass", np.float32),
            ("vx", np.float32),
            ("vy", np.float32),
            ("vz", np.float32),
            ("x", np.float64),  # using double precision for coordinates
            ("y", np.float64),
            ("z", np.float64),
            ("paraspin", np.float32),
            ("sigv", np.float32),
            ("rotation", np.float32),
            ("radius", np.float32),
            ("q", np.float32),
            ("s", np.float32),
            ("kw", np.float32),
            ("ang_mom", np.float32, (3,)),  # array of 3 floats for angular momentum
            ("v", np.float32, (3, 3)),  # 3x3 matrix for velocities or other properties
            ("pad1", "byte", 4),
        ]
    )

    with open(haloqcatpath, "rb") as fp:
        haloprop = np.fromfile(fp, dtype=dtype_halo_prop, count=-1)
    
    return haloprop


def read_halomemcat(halomempath):
    # define the basicparticletype structure
    dtype_basicparticletype = np.dtype(
        [
            ("x", np.float64),  # double precision float
            ("y", np.float64),
            ("z", np.float64),
            ("vx", np.float32),
            ("vy", np.float32),
            ("vz", np.float32),
            ("pad1", "byte", 4),
            ("indx", np.int64),
        ]
    )

    # open the member particle file
    with open(halomempath, "rb") as fp:
        particles = np.fromfile(fp, dtype=dtype_basicparticletype, count=-1)
        
    return particles


class Halos(object):
    def __init__(self, halocat, haloqcat, halomemcat, snapshot_props, initial):
        
        self.num_halos = len(halocat)
        self.hid = haloqcat["hid"] # halo id
        self.timestep = snapshot_props["istep"] # timestep index of the snapshot
        self.redshift = (snapshot_props["amax"]/snapshot_props["anow"]) - 1
        self.mass = haloqcat["mass"]
        self.position = halocat[["x", "y", "z"]]
        self.velocity = halocat[["vx", "vy", "vz"]]
        self.mbp = haloqcat["mbp"] # most bound particle of this halo
        
        # assign hid to particles
        self.num_ptls = halocat['np'].astype(int)
        self.memptlhid = np.repeat(np.arange(self.num_halos), self.num_ptls)
        self.member_particles = halomemcat
        
        # mbps tracked from the merged halos are not known initially
        self.initial = initial
        self.mbps_stacked = None # array of (mbp index, timestep index)

    def match_halos_previous_timestep(self, halos_previous):
        
        ptls = self.member_particles['indx'] # particles in the halos
        mbps = halos_previous.mbp # mbps of the halos in the previous timestep
        
        ptls_argsort = np.argsort(ptls)
        mbps_argsort = np.argsort(mbps)

        matches = np.isin(ptls[ptls_argsort], mbps[mbps_argsort])
        hptlid = np.nonzero(matches)[0]

        matches = np.isin(mbps[mbps_argsort], ptls[ptls_argsort])
        hmbpid = np.nonzero(matches)[0]

        self.hid_matched_now = self.memptlhid[ptls_argsort[hptlid]]
        self.hid_matched_prev = halos_previous.hid[mbps_argsort[hmbpid]]
        
        return self.hid_matched_now, self.hid_matched_prev
    
    def aggregate_mbps(self, halos_previous):
        if self.initial:
            self.mbps_stacked = np.column_stack((self.mbp, np.full_like(self.mbp, self.timestep, dtype=int)))
        else:
            # stack the mbps of newly-formed halos in this step
            new_halos = np.setdiff1d(self.hid, np.unique(self.hid_matched_now))
            new_mbps = self.mbp[np.isin(self.hid, new_halos)]
            self.mbps_stacked = np.vstack((halos_previous.mbps_stacked, [new_mbps, np.full_like(new_mbps, self.timestep)]))
            
        return self.mbps_stacked


    

        