-- Remove all MS2 scores which we do not use in the tests
delete from spectra_candidate_scores where participant not in ('MetFrag_2.4.5__8afe4a14', 'IOKR__696a17f3');

-- Remove Classyfire data
drop table classyfire_data;

-- Remove the molecular descriptors
drop table descriptors_data;
drop table descriptors_meta;

-- Remove the performance table
drop table performance;

-- We do not need the RankSVM's preference scores for the tests --> drop it
drop table preference_scores_data;
drop table preference_scores_meta;
drop table prefmodel_challenges_sample;

-- Remove the sample correspondence of the challenge spectra
drop table challenges_spectra_sample;

-- Remove no needed fingerprint data
create table fingerprints_data_dg_tmp
(
	molecule VARCHAR
		primary key
		references molecules,
	substructure_count VARCHAR,
	iokr_fps__positive VARCHAR
);

insert into fingerprints_data_dg_tmp(molecule, substructure_count, iokr_fps__positive)
    select molecule, substructure_count, iokr_fps__positive from fingerprints_data;

drop table fingerprints_data;

alter table fingerprints_data_dg_tmp rename to fingerprints_data;

delete from fingerprints_meta where name not in ('substructure_count', 'iokr_fps__positive');

-- Remove information to which mixture each spectra belongs
drop table spectra_mix;

-- Remove not used molecule information
create table molecules_dg_tmp
(
	inchi VARCHAR
		primary key,
	inchi2D VARCHAR not null,
	inchikey VARCHAR not null,
	inchikey1 VARCHAR not null
);

insert into molecules_dg_tmp(inchi, inchi2D, inchikey, inchikey1) select inchi, inchi2D, inchikey, inchikey1 from molecules;

drop table molecules;

alter table molecules_dg_tmp rename to molecules;

create index molecules_inchi2D_index
	on molecules (inchi2D);

create index molecules_inchikey1_index
	on molecules (inchikey1);

create index molecules_inchikey_index
	on molecules (inchikey);

-- Remove some MS2 spectra --> here 500
delete from spectra
    where name in (
        select name from spectra
            where name not in ('Challenge-016', 'EAX030601', 'Challenge-019', 'EAX034002', 'Challenge-017',
                               'Challenge-018', 'Challenge-019', 'Challenge-001')
            order by random()
            limit 500
    );

-- Remove now unused candidate scores and candidate associations
delete from spectra_candidate_scores
    where spectrum not in (select name from spectra);

delete from candidates_spectra
    where spectrum not in (select name from spectra);

-- Remove molecules that appear nowhere anymore
-- --> Start with the fingerprints
delete from fingerprints_data
    where molecule not in (select candidate from candidates_spectra);

-- --> Then go to the molecules
delete from molecules
    where inchi not in (select candidate from candidates_spectra);

-- Do not forget to ran VACUUM after removing all the data.
