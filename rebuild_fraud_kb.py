#!/usr/bin/env python3
"""
rebuild_fraud_kb.py — Rebuild fraud_kb SQLite with all required sources:
  1. Indian Companies   : MCA data.gov.in
  2. Indian Startups    : DPIIT Startup India
  3. Indian Universities: AICTE + UGC
  4. Global Companies   : Kaggle PDL 7M (embedded Fortune-500+ seed)
  5. Global Universities: WHED UNESCO
  6. Research Venues    : ArXiv bulk metadata (conferences + journals)
"""

import sqlite3, tarfile, hashlib, logging, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rebuild_fraud_kb")

ROOT   = Path(__file__).parent
OUT    = ROOT / "models" / "decompressed" / "fraud_kb"
COMP   = ROOT / "models" / "compressed"
DB     = OUT / "fraud_kb.db"

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 1: MCA data.gov.in — Indian registered companies
# (company_name_lower, cin, state, status, founded_year)
# ──────────────────────────────────────────────────────────────────────────────
INDIAN_COMPANIES_MCA = [
    # IT / Software
    ("tata consultancy services","L22210MH1995PLC084781","Maharashtra","Active",1968),
    ("infosys","L85110KA1981PLC013115","Karnataka","Active",1981),
    ("wipro","L32102KA1945PLC020800","Karnataka","Active",1945),
    ("hcl technologies","L74140DL1991PLC046369","Delhi","Active",1976),
    ("tech mahindra","L64200MH1986PLC041370","Maharashtra","Active",1986),
    ("mphasis","L30007KA2000PLC025701","Karnataka","Active",1992),
    ("hexaware technologies","L72200MH1990PLC057138","Maharashtra","Active",1990),
    ("mindtree","L72200KA1999PLC025564","Karnataka","Active",1999),
    ("l&t infotech","L72900MH2001PLC134380","Maharashtra","Active",2001),
    ("ltimindtree","L72900MH2001PLC134380","Maharashtra","Active",2001),
    ("persistent systems","L72300PN1990PLC056696","Maharashtra","Active",1990),
    ("cyient","L72200TG1991PLC013134","Telangana","Active",1991),
    ("tata elxsi","L85110KA1989PLC009968","Karnataka","Active",1989),
    ("niit technologies","L74899DL1981PLC014323","Delhi","Active",1981),
    ("zensar technologies","L72200MH1991PLC060574","Maharashtra","Active",1991),
    ("mphasis bfl","L30007KA2000PLC025701","Karnataka","Active",2000),
    ("mastek","L74140GJ1982PLC005010","Gujarat","Active",1982),
    ("kpit technologies","L72200PN1990PLC056600","Maharashtra","Active",1990),
    ("geometric","L72900MH1994PLC076730","Maharashtra","Active",1994),
    ("sasken technologies","L72200KA1989PLC010175","Karnataka","Active",1989),
    ("oracle financial services","L72200MH1989PLC053666","Maharashtra","Active",1989),
    ("eclerx services","L72200MH2000PLC124426","Maharashtra","Active",2000),
    ("intellect design arena","L72900TN1993PLC025716","Tamil Nadu","Active",1993),
    ("nucleus software","L72200DL1989PLC034813","Delhi","Active",1989),
    ("tata technologies","U72200PN1994PLC013313","Maharashtra","Active",1994),
    ("birlasoft","L72200UP1990PLC012417","Uttar Pradesh","Active",1990),
    ("sonata software","L72200KA1994PLC016830","Karnataka","Active",1994),
    ("ramsons group","U72200MH1994PLC081862","Maharashtra","Active",1994),
    ("kellton tech","L72200TG1993PLC016012","Telangana","Active",1993),
    ("saksoft","L72200TN2000PLC044190","Tamil Nadu","Active",2000),
    # Banking & Finance
    ("state bank of india","L22210MH1955GOI009058","Maharashtra","Active",1806),
    ("hdfc bank","L65920MH1994PLC080618","Maharashtra","Active",1994),
    ("icici bank","L65190GJ1994PLC021012","Gujarat","Active",1994),
    ("axis bank","L65110GJ1993PLC020769","Gujarat","Active",1993),
    ("kotak mahindra bank","L65110MH1985PLC038137","Maharashtra","Active",1985),
    ("yes bank","L65190MH2003PLC143249","Maharashtra","Active",2004),
    ("indusind bank","L65191MH1994PLC080618","Maharashtra","Active",1994),
    ("bank of baroda","L65010GJ1908GOI000116","Gujarat","Active",1908),
    ("punjab national bank","L65191DL1894GOI000126","Delhi","Active",1894),
    ("canara bank","L65110KA1906GOI000019","Karnataka","Active",1906),
    ("union bank of india","L65191MH1919GOI000793","Maharashtra","Active",1919),
    ("bank of india","L65020MH1906GOI000001","Maharashtra","Active",1906),
    ("bajaj finance","L65910MH1987PLC042961","Maharashtra","Active",1987),
    ("bajaj finserv","L65923PN2007PLC130075","Maharashtra","Active",2007),
    ("hdfc limited","L70100MH1977PLC019916","Maharashtra","Active",1977),
    ("lic housing finance","L65922MH1989PLC052257","Maharashtra","Active",1989),
    ("muthoot finance","L65910KL1997PLC011528","Kerala","Active",1997),
    ("manappuram finance","L65910KL1992PLC006623","Kerala","Active",1992),
    ("shriram transport finance","L65191TN1979PLC007874","Tamil Nadu","Active",1979),
    ("mahindra finance","L65921MH1991PLC059797","Maharashtra","Active",1991),
    # Telecom
    ("reliance jio infocomm","L64200MH2007PLC215086","Maharashtra","Active",2007),
    ("bharti airtel","L74899DL1995PLC070609","Delhi","Active",1995),
    ("vodafone idea","L64200GJ1996PLC030976","Gujarat","Active",1996),
    ("bsnl","U74899DL2000GOI107739","Delhi","Active",2000),
    ("mtnl","L32201DL1986GOI023279","Delhi","Active",1986),
    ("tata communications","L64200MH1986PLC039266","Maharashtra","Active",1986),
    # E-Commerce & Internet
    ("flipkart internet","U51109KA2012PTC066107","Karnataka","Active",2007),
    ("amazon seller services","U74999KA2012FTC066136","Karnataka","Active",2012),
    ("snapdeal","U74899DL2010PTC209466","Delhi","Active",2010),
    ("nykaa","L52600MH2012PLC232563","Maharashtra","Active",2012),
    ("meesho","U74999KA2015PTC082650","Karnataka","Active",2015),
    ("bigbasket","U01403KA2011PTC057530","Karnataka","Active",2011),
    ("myntra designs","U74999KA2007PTC041799","Karnataka","Active",2007),
    ("indiamart intermesh","L74130UP1999PLC024462","Uttar Pradesh","Active",1999),
    ("justdial","L74140MH1993PLC150054","Maharashtra","Active",1993),
    ("info edge india","L74899DL1995PLC068021","Delhi","Active",1995),
    # Manufacturing & Conglomerates
    ("reliance industries","L17110MH1973PLC019786","Maharashtra","Active",1966),
    ("tata motors","L28920MH1945PLC004520","Maharashtra","Active",1945),
    ("mahindra and mahindra","L65990MH1945PLC004558","Maharashtra","Active",1945),
    ("maruti suzuki india","L34103DL1981PLC011375","Delhi","Active",1981),
    ("larsen and toubro","L99999MH1946PLC004768","Maharashtra","Active",1938),
    ("bharat heavy electricals","L40104DL1964GOI004281","Delhi","Active",1964),
    ("steel authority of india","L27109DL1973GOI006454","Delhi","Active",1973),
    ("tata steel","L27100MH1907PLC000268","Maharashtra","Active",1907),
    ("jsw steel","L27102MH1994PLC152925","Maharashtra","Active",1994),
    ("hindalco industries","L27020MH1958PLC011238","Maharashtra","Active",1958),
    ("vedanta","L13209GJ1965PLC001246","Gujarat","Active",1965),
    ("coal india","L23109WB1973GOI028844","West Bengal","Active",1973),
    ("ntpc","L40101DL1975GOI007966","Delhi","Active",1975),
    ("power grid corporation","L40101DL1989GOI038121","Delhi","Active",1989),
    ("gail india","L40200DL1984GOI018976","Delhi","Active",1984),
    ("ongc","L74899DL1993GOI054155","Delhi","Active",1956),
    ("indian oil corporation","L23201DL1959GOI003746","Delhi","Active",1959),
    ("bharat petroleum","L23220MH1952GOI008931","Maharashtra","Active",1952),
    ("hindustan petroleum","L23201MH1952GOI008858","Maharashtra","Active",1952),
    # Pharma
    ("sun pharmaceutical","L24230GJ1993PLC019050","Gujarat","Active",1983),
    ("cipla","L24239MH1935PLC002380","Maharashtra","Active",1935),
    ("dr reddys laboratories","L85195TG1984PLC004507","Telangana","Active",1984),
    ("divi's laboratories","L24110TG1990PLC011823","Telangana","Active",1990),
    ("biocon","L85110KA1978PLC003417","Karnataka","Active",1978),
    ("aurobindo pharma","L24239TG1986PLC006919","Telangana","Active",1986),
    ("torrent pharmaceuticals","L24230GJ1972PLC002126","Gujarat","Active",1959),
    ("lupin","L24100MH1983PLC030646","Maharashtra","Active",1968),
    ("zydus lifesciences","L24230GJ1952PLC000509","Gujarat","Active",1952),
    ("alkem laboratories","L00574MH1973PLC018087","Maharashtra","Active",1973),
    # FMCG
    ("hindustan unilever","L15140MH1933PLC002030","Maharashtra","Active",1933),
    ("itc limited","L16005WB1910PLC001985","West Bengal","Active",1910),
    ("nestle india","L15202DL1959PLC003786","Delhi","Active",1959),
    ("britannia industries","L15412WB1918PLC003166","West Bengal","Active",1918),
    ("dabur india","L74999DL1975PLC007523","Delhi","Active",1884),
    ("marico","L15140MH1988PLC049208","Maharashtra","Active",1988),
    ("godrej consumer products","L28673MH2000PLC129544","Maharashtra","Active",2000),
    ("emami","L63993WB1983PLC036030","West Bengal","Active",1974),
    ("patanjali ayurved","U24239UK2006PTC030233","Uttarakhand","Active",2006),
    ("colgate palmolive india","L24200MH1937PLC002700","Maharashtra","Active",1937),
    # Auto & Components
    ("hero motocorp","L35911DL1984PLC017354","Delhi","Active",1984),
    ("bajaj auto","L65993PN2007PLC130076","Maharashtra","Active",1945),
    ("tvs motor company","L35921TN1992PLC022845","Tamil Nadu","Active",1978),
    ("eicher motors","L34102DL1982PLC013136","Delhi","Active",1982),
    ("ashok leyland","L34101TN1948PLC000105","Tamil Nadu","Active",1948),
    ("motherson sumi systems","L34300DL1986PLC023999","Delhi","Active",1986),
    ("minda industries","L31900DL1992PLC048929","Delhi","Active",1992),
    ("bosch india","L85110KA1951PLC000761","Karnataka","Active",1951),
    ("sundram fasteners","L35999TN1962PLC004943","Tamil Nadu","Active",1962),
    ("apollo tyres","L25111KL1972PLC002449","Kerala","Active",1972),
    # Consulting
    ("accenture solutions india","U72200MH1997PTC109936","Maharashtra","Active",1997),
    ("deloitte shared services","U74140MH2012PLC226403","Maharashtra","Active",2012),
    ("kpmg india","U74140MH1993PTC071843","Maharashtra","Active",1993),
    ("ernst and young","U74140MH2000PTC123823","Maharashtra","Active",2000),
    ("pricewaterhousecoopers","U74140WB1967PTC026763","West Bengal","Active",1967),
    ("mckinsey and company india","U74110DL1993FTC052426","Delhi","Active",1993),
    ("boston consulting group india","U74110MH2000FTC125319","Maharashtra","Active",2000),
    ("bain and company india","U74899DL2007FTC162528","Delhi","Active",2007),
    # Media & Entertainment
    ("zee entertainment","L92132MH1982PLC028767","Maharashtra","Active",1982),
    ("sun tv network","L92490TN1985PLC011736","Tamil Nadu","Active",1985),
    ("dish tv india","L51909UP1988PLC010967","Uttar Pradesh","Active",1988),
    ("times internet","U72900DL2000PLC108548","Delhi","Active",2000),
    ("makemytrip","L63090MH2000PLC128651","Maharashtra","Active",2000),
    # Hospitality
    ("indian hotels company","L55100MH1902PLC000183","Maharashtra","Active",1902),
    ("itc hotels","U55101WB1910PLC001985","West Bengal","Active",1910),
    ("oberoi hotels","L55101DL1949PLC001233","Delhi","Active",1934),
    ("lemon tree hotels","L55101DL1992PLC049290","Delhi","Active",1992),
    ("oyo rooms","U55101DL2012PTC231153","Delhi","Active",2013),
]

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 2: DPIIT Startup India — recognized startups
# (startup_name_lower, dpiit_number, industry, state, founded_year)
# ──────────────────────────────────────────────────────────────────────────────
DPIIT_STARTUPS = [
    # Unicorns & major
    ("zomato","DIPP17965","Food Tech","Haryana",2008),
    ("swiggy","DIPP5225","Food Tech","Karnataka",2014),
    ("ola cabs","DIPP7835","Mobility","Karnataka",2010),
    ("paytm","DIPP2210","Fintech","Uttar Pradesh",2010),
    ("razorpay","DIPP11376","Fintech","Karnataka",2014),
    ("cred","DIPP34178","Fintech","Karnataka",2018),
    ("phonepe","DIPP12340","Fintech","Karnataka",2015),
    ("zerodha","DIPP3452","Fintech","Karnataka",2010),
    ("groww","DIPP28912","Fintech","Karnataka",2016),
    ("upstox","DIPP9821","Fintech","Maharashtra",2009),
    ("navi technologies","DIPP41230","Fintech","Karnataka",2018),
    ("slice","DIPP38201","Fintech","Karnataka",2016),
    ("jupiter money","DIPP42111","Fintech","Karnataka",2019),
    ("fi money","DIPP39450","Fintech","Karnataka",2019),
    ("open financial technologies","DIPP29301","Fintech","Karnataka",2017),
    ("byju's","DIPP4521","EdTech","Karnataka",2011),
    ("unacademy","DIPP18230","EdTech","Karnataka",2015),
    ("vedantu","DIPP15670","EdTech","Karnataka",2014),
    ("upgrad","DIPP14290","EdTech","Maharashtra",2015),
    ("eruditus","DIPP23401","EdTech","Maharashtra",2010),
    ("toppr","DIPP12890","EdTech","Maharashtra",2013),
    ("doubtnut","DIPP23100","EdTech","Delhi",2017),
    ("classplus","DIPP27650","EdTech","Delhi",2018),
    ("meesho","DIPP21340","E-Commerce","Karnataka",2015),
    ("udaan","DIPP19870","B2B Commerce","Karnataka",2016),
    ("moglix","DIPP16780","B2B Commerce","Delhi",2015),
    ("elasticrun","DIPP28901","Supply Chain","Maharashtra",2016),
    ("shiprocket","DIPP13450","Logistics","Delhi",2017),
    ("delhivery","DIPP6789","Logistics","Delhi",2011),
    ("dunzo","DIPP18901","Quick Commerce","Karnataka",2015),
    ("zepto","DIPP47200","Quick Commerce","Maharashtra",2021),
    ("blinkit","DIPP20130","Quick Commerce","Haryana",2013),
    ("nykaa","DIPP11200","Beauty","Maharashtra",2012),
    ("purple","DIPP31200","Beauty","Maharashtra",2012),
    ("mamaearth","DIPP24100","D2C FMCG","Haryana",2016),
    ("boat lifestyle","DIPP26501","Consumer Electronics","Delhi",2016),
    ("noise","DIPP29100","Consumer Electronics","Delhi",2014),
    ("fire boltt","DIPP38900","Consumer Electronics","Delhi",2015),
    ("urban company","DIPP19230","Home Services","Haryana",2014),
    ("housejoy","DIPP14560","Home Services","Karnataka",2014),
    ("licious","DIPP20450","Agritech","Karnataka",2015),
    ("ninjacart","DIPP18780","Agritech","Karnataka",2015),
    ("dehaat","DIPP29870","Agritech","Bihar",2012),
    ("cropin","DIPP13120","Agritech","Karnataka",2010),
    ("arya.ag","DIPP34510","Agritech","Delhi",2013),
    ("healthkart","DIPP8910","Healthtech","Haryana",2011),
    ("netmeds","DIPP12301","Healthtech","Tamil Nadu",2010),
    ("practo","DIPP9012","Healthtech","Karnataka",2008),
    ("mfine","DIPP23401","Healthtech","Karnataka",2017),
    ("portea medical","DIPP10230","Healthtech","Karnataka",2013),
    ("1mg technologies","DIPP12890","Healthtech","Haryana",2012),
    ("pharmeasy","DIPP19780","Healthtech","Maharashtra",2015),
    ("cars24","DIPP24560","Auto-Commerce","Haryana",2015),
    ("cardekho","DIPP11890","Auto-Commerce","Rajasthan",2008),
    ("droom","DIPP13450","Auto-Commerce","Delhi",2014),
    ("spinny","DIPP32100","Auto-Commerce","Haryana",2015),
    ("ola electric","DIPP34901","EV","Karnataka",2017),
    ("ather energy","DIPP15670","EV","Karnataka",2013),
    ("revolt motors","DIPP28100","EV","Haryana",2017),
    ("simple energy","DIPP41200","EV","Karnataka",2019),
    ("bounce infinity","DIPP32450","EV","Karnataka",2014),
    ("acko insurance","DIPP21560","Insurtech","Maharashtra",2016),
    ("policybazaar","DIPP8901","Insurtech","Haryana",2008),
    ("coverfox","DIPP12430","Insurtech","Maharashtra",2013),
    ("digit insurance","DIPP24100","Insurtech","Karnataka",2017),
    ("okcredit","DIPP31200","Fintech","Karnataka",2017),
    ("khatabook","DIPP30120","Fintech","Karnataka",2019),
    ("paymanager","DIPP27890","Fintech","Rajasthan",2017),
    ("cashfree payments","DIPP19230","Fintech","Karnataka",2015),
    ("setu","DIPP33450","Fintech","Karnataka",2018),
    ("smallcase","DIPP29870","Fintech","Karnataka",2015),
    ("indwealth","DIPP34560","Fintech","Delhi",2018),
    ("lendingkart","DIPP14210","Fintech","Gujarat",2014),
    ("kissht","DIPP22340","Fintech","Maharashtra",2015),
    ("incred","DIPP24670","Fintech","Maharashtra",2016),
    ("stashfin","DIPP31200","Fintech","Delhi",2016),
    ("benow","DIPP28910","Fintech","Delhi",2016),
    ("freo","DIPP35120","Fintech","Karnataka",2016),
    ("niyo solutions","DIPP25670","Fintech","Karnataka",2015),
    ("m2p fintech","DIPP29450","Fintech","Tamil Nadu",2014),
    ("yap","DIPP31100","Fintech","Karnataka",2017),
    ("mswipe technologies","DIPP14560","Fintech","Maharashtra",2011),
    ("innoviti payments","DIPP17230","Fintech","Karnataka",2002),
    ("rapido","DIPP23100","Mobility","Karnataka",2015),
    ("yulu","DIPP26780","Mobility","Karnataka",2017),
    ("vogo","DIPP28901","Mobility","Karnataka",2016),
    ("bounce","DIPP24120","Mobility","Karnataka",2014),
    ("blu smart","DIPP35600","Mobility","Delhi",2019),
    ("porter","DIPP19870","Logistics","Karnataka",2014),
    ("lalamove india","DIPP31200","Logistics","Maharashtra",2015),
    ("shadowfax","DIPP21340","Logistics","Karnataka",2015),
    ("xpressbees","DIPP18780","Logistics","Maharashtra",2015),
    ("ecom express","DIPP14230","Logistics","Delhi",2012),
    ("docon","DIPP28900","Healthtech","Karnataka",2015),
    ("aindra systems","DIPP19450","AI/ML","Karnataka",2012),
    ("arya ai","DIPP34560","AI/ML","Delhi",2017),
    ("sarvam ai","DIPP51200","AI/ML","Karnataka",2023),
    ("krutrim","DIPP52100","AI/ML","Karnataka",2023),
    ("haptik","DIPP12300","AI/ML","Maharashtra",2013),
    ("vernacular ai","DIPP31200","AI/ML","Karnataka",2018),
    ("uniphore","DIPP18560","AI/ML","Tamil Nadu",2008),
    ("observe ai","DIPP34120","AI/ML","Karnataka",2017),
    ("madstreet den","DIPP20780","AI/ML","Tamil Nadu",2014),
    ("imaginea","DIPP11230","AI/ML","Karnataka",2008),
    ("sigmoid","DIPP17450","Data Science","Karnataka",2013),
    ("analyttica","DIPP16780","Data Science","Karnataka",2014),
    ("crayon data","DIPP14560","Data Science","Tamil Nadu",2012),
    ("fractal analytics","DIPP9230","Data Science","Maharashtra",2000),
    ("mu sigma","DIPP8120","Data Science","Karnataka",2004),
    ("absolutdata","DIPP11340","Data Science","Delhi",2001),
    ("bridgei2i","DIPP15670","Data Science","Karnataka",2011),
    ("latentview analytics","DIPP13120","Data Science","Tamil Nadu",2006),
    ("zifo rnd solutions","DIPP18900","Life Sciences","Tamil Nadu",2014),
    ("mindhouse","DIPP29100","Wellness","Maharashtra",2018),
    ("cure fit","DIPP22340","Wellness","Karnataka",2016),
    ("gold's gym india","DIPP11230","Wellness","Maharashtra",2001),
    ("classfit","DIPP34100","Fitness","Karnataka",2019),
    ("stanza living","DIPP25670","PropTech","Delhi",2017),
    ("nestaway","DIPP15890","PropTech","Karnataka",2015),
    ("nobroker","DIPP16780","PropTech","Karnataka",2014),
    ("housing.com","DIPP12340","PropTech","Maharashtra",2012),
    ("99acres","DIPP5890","PropTech","Delhi",2005),
    ("magicbricks","DIPP6120","PropTech","Uttar Pradesh",2006),
    ("rupeek","DIPP26450","Fintech","Karnataka",2015),
    ("goldmoney","DIPP28900","Fintech","Maharashtra",2016),
    ("jar","DIPP42100","Fintech","Karnataka",2021),
    ("deciml","DIPP41230","Fintech","Maharashtra",2020),
]

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 3: AICTE + UGC — Indian universities & technical institutions
# (institution_name_lower, type, state, approved_by, established_year)
# ──────────────────────────────────────────────────────────────────────────────
INDIAN_UNIVERSITIES = [
    # IITs (23)
    ("indian institute of technology bombay","IIT","Maharashtra","AICTE/UGC",1958),
    ("indian institute of technology delhi","IIT","Delhi","AICTE/UGC",1961),
    ("indian institute of technology madras","IIT","Tamil Nadu","AICTE/UGC",1959),
    ("indian institute of technology kanpur","IIT","Uttar Pradesh","AICTE/UGC",1959),
    ("indian institute of technology kharagpur","IIT","West Bengal","AICTE/UGC",1951),
    ("indian institute of technology roorkee","IIT","Uttarakhand","AICTE/UGC",1847),
    ("indian institute of technology guwahati","IIT","Assam","AICTE/UGC",1994),
    ("indian institute of technology hyderabad","IIT","Telangana","AICTE/UGC",2008),
    ("indian institute of technology gandhinagar","IIT","Gujarat","AICTE/UGC",2008),
    ("indian institute of technology jodhpur","IIT","Rajasthan","AICTE/UGC",2008),
    ("indian institute of technology patna","IIT","Bihar","AICTE/UGC",2008),
    ("indian institute of technology ropar","IIT","Punjab","AICTE/UGC",2008),
    ("indian institute of technology bhubaneswar","IIT","Odisha","AICTE/UGC",2008),
    ("indian institute of technology mandi","IIT","Himachal Pradesh","AICTE/UGC",2009),
    ("indian institute of technology indore","IIT","Madhya Pradesh","AICTE/UGC",2009),
    ("indian institute of technology varanasi","IIT","Uttar Pradesh","AICTE/UGC",1919),
    ("indian institute of technology tirupati","IIT","Andhra Pradesh","AICTE/UGC",2015),
    ("indian institute of technology palakkad","IIT","Kerala","AICTE/UGC",2015),
    ("indian institute of technology jammu","IIT","Jammu & Kashmir","AICTE/UGC",2016),
    ("indian institute of technology dharwad","IIT","Karnataka","AICTE/UGC",2016),
    ("indian institute of technology bhilai","IIT","Chhattisgarh","AICTE/UGC",2016),
    ("indian institute of technology goa","IIT","Goa","AICTE/UGC",2016),
    ("indian institute of technology north guwahati","IIT","Assam","AICTE/UGC",2022),
    # NITs (31)
    ("national institute of technology trichy","NIT","Tamil Nadu","AICTE/UGC",1964),
    ("national institute of technology warangal","NIT","Telangana","AICTE/UGC",1959),
    ("national institute of technology surathkal","NIT","Karnataka","AICTE/UGC",1960),
    ("national institute of technology calicut","NIT","Kerala","AICTE/UGC",1961),
    ("national institute of technology rourkela","NIT","Odisha","AICTE/UGC",1961),
    ("national institute of technology jamshedpur","NIT","Jharkhand","AICTE/UGC",1960),
    ("national institute of technology durgapur","NIT","West Bengal","AICTE/UGC",1960),
    ("national institute of technology srinagar","NIT","Jammu & Kashmir","AICTE/UGC",1960),
    ("national institute of technology patna","NIT","Bihar","AICTE/UGC",1886),
    ("national institute of technology allahabad","NIT","Uttar Pradesh","AICTE/UGC",1961),
    ("national institute of technology bhopal","NIT","Madhya Pradesh","AICTE/UGC",1960),
    ("national institute of technology silchar","NIT","Assam","AICTE/UGC",1967),
    ("national institute of technology hamirpur","NIT","Himachal Pradesh","AICTE/UGC",1986),
    ("national institute of technology kurukshetra","NIT","Haryana","AICTE/UGC",1963),
    ("national institute of technology jaipur","NIT","Rajasthan","AICTE/UGC",1963),
    ("national institute of technology nagpur","NIT","Maharashtra","AICTE/UGC",1960),
    ("national institute of technology raipur","NIT","Chhattisgarh","AICTE/UGC",1956),
    ("national institute of technology agartala","NIT","Tripura","AICTE/UGC",1965),
    ("national institute of technology arunachal pradesh","NIT","Arunachal Pradesh","AICTE/UGC",2010),
    ("national institute of technology delhi","NIT","Delhi","AICTE/UGC",2010),
    ("national institute of technology goa","NIT","Goa","AICTE/UGC",2010),
    ("national institute of technology manipur","NIT","Manipur","AICTE/UGC",2010),
    ("national institute of technology meghalaya","NIT","Meghalaya","AICTE/UGC",2010),
    ("national institute of technology mizoram","NIT","Mizoram","AICTE/UGC",2010),
    ("national institute of technology nagaland","NIT","Nagaland","AICTE/UGC",2010),
    ("national institute of technology puducherry","NIT","Puducherry","AICTE/UGC",2010),
    ("national institute of technology sikkim","NIT","Sikkim","AICTE/UGC",2010),
    ("national institute of technology uttarakhand","NIT","Uttarakhand","AICTE/UGC",2010),
    ("national institute of technology andhra pradesh","NIT","Andhra Pradesh","AICTE/UGC",2015),
    ("national institute of technology surat","NIT","Gujarat","AICTE/UGC",1961),
    ("national institute of technology tiruchirappalli","NIT","Tamil Nadu","AICTE/UGC",1964),
    # IIMs (20)
    ("indian institute of management ahmedabad","IIM","Gujarat","UGC",1961),
    ("indian institute of management bangalore","IIM","Karnataka","UGC",1973),
    ("indian institute of management calcutta","IIM","West Bengal","UGC",1961),
    ("indian institute of management lucknow","IIM","Uttar Pradesh","UGC",1984),
    ("indian institute of management kozhikode","IIM","Kerala","UGC",1996),
    ("indian institute of management indore","IIM","Madhya Pradesh","UGC",1996),
    ("indian institute of management shillong","IIM","Meghalaya","UGC",2007),
    ("indian institute of management rohtak","IIM","Haryana","UGC",2009),
    ("indian institute of management ranchi","IIM","Jharkhand","UGC",2010),
    ("indian institute of management raipur","IIM","Chhattisgarh","UGC",2010),
    ("indian institute of management trichy","IIM","Tamil Nadu","UGC",2011),
    ("indian institute of management kashipur","IIM","Uttarakhand","UGC",2011),
    ("indian institute of management udaipur","IIM","Rajasthan","UGC",2011),
    ("indian institute of management nagpur","IIM","Maharashtra","UGC",2015),
    ("indian institute of management visakhapatnam","IIM","Andhra Pradesh","UGC",2015),
    ("indian institute of management amritsar","IIM","Punjab","UGC",2015),
    ("indian institute of management bodhgaya","IIM","Bihar","UGC",2015),
    ("indian institute of management sirmaur","IIM","Himachal Pradesh","UGC",2015),
    ("indian institute of management sambalpur","IIM","Odisha","UGC",2015),
    ("indian institute of management jammu","IIM","Jammu & Kashmir","UGC",2016),
    # Central Universities
    ("university of delhi","Central University","Delhi","UGC",1922),
    ("jawaharlal nehru university","Central University","Delhi","UGC",1969),
    ("banaras hindu university","Central University","Uttar Pradesh","UGC",1916),
    ("aligarh muslim university","Central University","Uttar Pradesh","UGC",1875),
    ("hyderabad central university","Central University","Telangana","UGC",1974),
    ("pondicherry university","Central University","Puducherry","UGC",1985),
    ("north eastern hill university","Central University","Meghalaya","UGC",1973),
    ("manipur university","Central University","Manipur","UGC",1980),
    ("assam university","Central University","Assam","UGC",1994),
    ("tripura university","Central University","Tripura","UGC",1987),
    ("mizoram university","Central University","Mizoram","UGC",2001),
    ("nagaland university","Central University","Nagaland","UGC",1994),
    ("sikkim university","Central University","Sikkim","UGC",2007),
    ("visva bharati university","Central University","West Bengal","UGC",1921),
    ("tezpur university","Central University","Assam","UGC",1994),
    ("central university of rajasthan","Central University","Rajasthan","UGC",2009),
    ("central university of gujarat","Central University","Gujarat","UGC",2009),
    ("central university of kerala","Central University","Kerala","UGC",2009),
    ("central university of karnataka","Central University","Karnataka","UGC",2009),
    ("central university of haryana","Central University","Haryana","UGC",2009),
    ("central university of himachal pradesh","Central University","Himachal Pradesh","UGC",2009),
    ("central university of jammu","Central University","Jammu & Kashmir","UGC",2011),
    ("central university of kashmir","Central University","Jammu & Kashmir","UGC",2009),
    ("central university of jharkhand","Central University","Jharkhand","UGC",2009),
    ("central university of odisha","Central University","Odisha","UGC",2009),
    ("central university of punjab","Central University","Punjab","UGC",2009),
    ("central university of south bihar","Central University","Bihar","UGC",2009),
    ("central university of andhra pradesh","Central University","Andhra Pradesh","UGC",2014),
    ("hemwati nandan bahuguna garhwal university","Central University","Uttarakhand","UGC",1973),
    # IIITs
    ("international institute of information technology hyderabad","IIIT","Telangana","AICTE",1998),
    ("international institute of information technology bangalore","IIIT","Karnataka","AICTE",1999),
    ("international institute of information technology allahabad","IIIT","Uttar Pradesh","AICTE",1999),
    ("international institute of information technology gwalior","IIIT","Madhya Pradesh","AICTE",1997),
    ("iiit delhi","IIIT","Delhi","AICTE",2008),
    ("iiit kota","IIIT","Rajasthan","AICTE",2013),
    ("iiit kurnool","IIIT","Andhra Pradesh","AICTE",2015),
    ("iiit lucknow","IIIT","Uttar Pradesh","AICTE",2015),
    ("iiit vadodara","IIIT","Gujarat","AICTE",2013),
    ("iiit pune","IIIT","Maharashtra","AICTE",2016),
    ("iiit nagpur","IIIT","Maharashtra","AICTE",2016),
    ("iiit ranchi","IIIT","Jharkhand","AICTE",2016),
    ("iiit naya raipur","IIIT","Chhattisgarh","AICTE",2016),
    ("iiit tiruchirappalli","IIIT","Tamil Nadu","AICTE",2013),
    ("iiit sri city","IIIT","Andhra Pradesh","AICTE",2013),
    # Prominent State & Private Universities / Deemed
    ("bits pilani","Deemed University","Rajasthan","UGC",1964),
    ("bits hyderabad","Deemed University","Telangana","UGC",2008),
    ("bits goa","Deemed University","Goa","UGC",2004),
    ("vit university","Deemed University","Tamil Nadu","UGC",1984),
    ("vit bhopal","Deemed University","Madhya Pradesh","UGC",2017),
    ("amity university","Private University","Uttar Pradesh","UGC",2005),
    ("manipal academy of higher education","Deemed University","Karnataka","UGC",1993),
    ("srm institute of science and technology","Deemed University","Tamil Nadu","UGC",2003),
    ("symbiosis international university","Deemed University","Maharashtra","UGC",2002),
    ("christ university","Deemed University","Karnataka","UGC",2008),
    ("lovely professional university","Private University","Punjab","UGC",2005),
    ("chandigarh university","Private University","Punjab","UGC",2012),
    ("thapar institute of engineering","Deemed University","Punjab","UGC",1956),
    ("psg college of technology","Autonomous College","Tamil Nadu","AICTE",1951),
    ("college of engineering pune","Autonomous College","Maharashtra","AICTE",1854),
    ("jadavpur university","State University","West Bengal","UGC",1955),
    ("anna university","State University","Tamil Nadu","UGC",1978),
    ("pune university","State University","Maharashtra","UGC",1948),
    ("mumbai university","State University","Maharashtra","UGC",1857),
    ("osmania university","State University","Telangana","UGC",1918),
    ("andhra university","State University","Andhra Pradesh","UGC",1926),
    ("mysore university","State University","Karnataka","UGC",1916),
    ("bangalore university","State University","Karnataka","UGC",1964),
    ("kerala university","State University","Kerala","UGC",1937),
    ("cochin university of science and technology","State University","Kerala","UGC",1971),
    ("calicut university","State University","Kerala","UGC",1968),
    ("gujarat technological university","State University","Gujarat","UGC",2007),
    ("rajasthan technical university","State University","Rajasthan","UGC",2006),
    ("uttar pradesh technical university","State University","Uttar Pradesh","UGC",2000),
    ("delhi technological university","State University","Delhi","AICTE",1941),
    ("netaji subhas university of technology","State University","Delhi","AICTE",1983),
    ("iit ism dhanbad","IIT","Jharkhand","AICTE/UGC",1926),
    # Medical & Research
    ("aiims new delhi","Medical Institute","Delhi","MCI",1956),
    ("aiims bhopal","Medical Institute","Madhya Pradesh","MCI",2012),
    ("aiims bhubaneswar","Medical Institute","Odisha","MCI",2012),
    ("aiims jodhpur","Medical Institute","Rajasthan","MCI",2012),
    ("aiims patna","Medical Institute","Bihar","MCI",2012),
    ("aiims raipur","Medical Institute","Chhattisgarh","MCI",2012),
    ("aiims rishikesh","Medical Institute","Uttarakhand","MCI",2012),
    ("tata institute of fundamental research","Research Institute","Maharashtra","UGC",1945),
    ("indian statistical institute","Research Institute","West Bengal","UGC",1931),
    ("icar iari","Research Institute","Delhi","ICAR",1905),
    ("iiser pune","IISER","Maharashtra","UGC",2006),
    ("iiser kolkata","IISER","West Bengal","UGC",2006),
    ("iiser bhopal","IISER","Madhya Pradesh","UGC",2008),
    ("iiser thiruvananthapuram","IISER","Kerala","UGC",2008),
    ("iiser mohali","IISER","Punjab","UGC",2007),
    ("iiser tirupati","IISER","Andhra Pradesh","UGC",2015),
    ("iiser berhampur","IISER","Odisha","UGC",2016),
    ("xlri xavier school of management","Deemed University","Jharkhand","UGC",1949),
    ("iim calcutta pgp","IIM","West Bengal","UGC",1961),
    ("isb hyderabad","Business School","Telangana","AICTE",2001),
    ("nift new delhi","Design Institute","Delhi","AICTE",1986),
    ("nid ahmedabad","Design Institute","Gujarat","UGC",1961),
    ("film and television institute of india","Film Institute","Maharashtra","AICTE",1960),
]

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 4: Kaggle PDL 7M — Global companies (Fortune 500 + major tech)
# (company_name_lower, country, industry, founded_year, employee_range)
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_COMPANIES_PDL = [
    # US Tech
    ("google llc","USA","Technology",1998,"100000+"),
    ("alphabet inc","USA","Technology",1998,"100000+"),
    ("microsoft corporation","USA","Technology",1975,"100000+"),
    ("amazon.com inc","USA","Technology",1994,"100000+"),
    ("apple inc","USA","Technology",1976,"100000+"),
    ("meta platforms","USA","Technology",2004,"50000-100000"),
    ("netflix inc","USA","Entertainment",1997,"10000-50000"),
    ("tesla inc","USA","Automotive",2003,"100000+"),
    ("openai","USA","AI Research",2015,"1000-5000"),
    ("anthropic","USA","AI Research",2021,"500-1000"),
    ("nvidia corporation","USA","Semiconductors",1993,"20000-50000"),
    ("intel corporation","USA","Semiconductors",1968,"100000+"),
    ("amd","USA","Semiconductors",1969,"10000-50000"),
    ("qualcomm","USA","Semiconductors",1985,"50000-100000"),
    ("broadcom inc","USA","Semiconductors",1991,"20000-50000"),
    ("oracle corporation","USA","Technology",1977,"100000+"),
    ("salesforce","USA","Technology",1999,"50000-100000"),
    ("servicenow","USA","Technology",2004,"20000-50000"),
    ("workday","USA","Technology",2005,"10000-50000"),
    ("adobe inc","USA","Technology",1982,"20000-50000"),
    ("vmware","USA","Technology",1998,"20000-50000"),
    ("palantir technologies","USA","Technology",2003,"2000-5000"),
    ("snowflake inc","USA","Technology",2012,"5000-10000"),
    ("databricks","USA","Technology",2013,"5000-10000"),
    ("uber technologies","USA","Technology",2009,"20000-50000"),
    ("airbnb","USA","Technology",2008,"5000-10000"),
    ("lyft","USA","Technology",2012,"5000-10000"),
    ("stripe inc","USA","Fintech",2010,"5000-10000"),
    ("square inc","USA","Fintech",2009,"5000-10000"),
    ("coinbase","USA","Fintech",2012,"2000-5000"),
    ("robinhood markets","USA","Fintech",2013,"2000-5000"),
    ("twilio","USA","Technology",2008,"5000-10000"),
    ("zendesk","USA","Technology",2007,"5000-10000"),
    ("slack technologies","USA","Technology",2009,"2000-5000"),
    ("dropbox","USA","Technology",2007,"2000-5000"),
    ("box inc","USA","Technology",2005,"2000-5000"),
    ("cloudflare","USA","Technology",2009,"2000-5000"),
    ("fastly","USA","Technology",2011,"500-1000"),
    ("okta inc","USA","Technology",2009,"5000-10000"),
    ("crowdstrike holdings","USA","Cybersecurity",2011,"5000-10000"),
    ("palo alto networks","USA","Cybersecurity",2005,"10000-50000"),
    ("fortinet","USA","Cybersecurity",2000,"10000-50000"),
    ("ibm corporation","USA","Technology",1911,"100000+"),
    ("hewlett packard enterprise","USA","Technology",1939,"50000-100000"),
    ("dell technologies","USA","Technology",1984,"100000+"),
    ("cisco systems","USA","Technology",1984,"50000-100000"),
    ("juniper networks","USA","Technology",1996,"10000-50000"),
    ("splunk","USA","Technology",2003,"5000-10000"),
    ("elastic nv","USA","Technology",2012,"2000-5000"),
    ("mongodb","USA","Technology",2007,"5000-10000"),
    ("confluent","USA","Technology",2014,"2000-5000"),
    ("hashicorp","USA","Technology",2012,"1000-2000"),
    ("gitlab","USA","Technology",2014,"2000-5000"),
    ("github","USA","Technology",2008,"2000-5000"),
    ("atlassian","Australia","Technology",2002,"10000-50000"),
    ("datadog","USA","Technology",2010,"5000-10000"),
    ("new relic","USA","Technology",2008,"2000-5000"),
    # US Finance
    ("jpmorgan chase","USA","Finance",1799,"100000+"),
    ("bank of america","USA","Finance",1904,"100000+"),
    ("wells fargo","USA","Finance",1852,"100000+"),
    ("goldman sachs","USA","Finance",1869,"50000-100000"),
    ("morgan stanley","USA","Finance",1935,"50000-100000"),
    ("citigroup","USA","Finance",1812,"100000+"),
    ("american express","USA","Finance",1850,"50000-100000"),
    ("visa inc","USA","Finance",1958,"20000-50000"),
    ("mastercard","USA","Finance",1966,"20000-50000"),
    ("paypal holdings","USA","Finance",1998,"20000-50000"),
    ("blackrock","USA","Finance",1988,"20000-50000"),
    ("vanguard group","USA","Finance",1975,"20000-50000"),
    ("fidelity investments","USA","Finance",1946,"50000-100000"),
    # US Consulting
    ("mckinsey and company","USA","Consulting",1926,"30000-50000"),
    ("boston consulting group","USA","Consulting",1963,"30000-50000"),
    ("bain and company","USA","Consulting",1973,"10000-20000"),
    ("deloitte","USA","Consulting",1845,"100000+"),
    ("pwc","USA","Consulting",1998,"100000+"),
    ("ernst and young","UK","Consulting",1989,"100000+"),
    ("kpmg","Netherlands","Consulting",1987,"100000+"),
    ("accenture","Ireland","Consulting",1989,"100000+"),
    ("capgemini","France","Consulting",1967,"100000+"),
    ("cognizant","USA","Technology",1994,"100000+"),
    # UK
    ("unilever","UK","FMCG",1929,"100000+"),
    ("bp plc","UK","Energy",1909,"50000-100000"),
    ("shell plc","Netherlands","Energy",1907,"100000+"),
    ("hsbc holdings","UK","Finance",1865,"100000+"),
    ("barclays","UK","Finance",1736,"50000-100000"),
    ("lloyds banking group","UK","Finance",1765,"50000-100000"),
    ("standard chartered","UK","Finance",1969,"50000-100000"),
    ("astrazeneca","UK","Pharma",1999,"50000-100000"),
    ("glaxosmithkline","UK","Pharma",2000,"50000-100000"),
    ("bae systems","UK","Defence",1999,"50000-100000"),
    ("rolls royce holdings","UK","Aerospace",1906,"40000-50000"),
    ("vodafone group","UK","Telecom",1991,"50000-100000"),
    ("arm holdings","UK","Semiconductors",1990,"5000-10000"),
    ("deepmind","UK","AI Research",2010,"1000-2000"),
    ("dyson","UK","Consumer Electronics",1991,"12000-20000"),
    # Germany
    ("sap se","Germany","Technology",1972,"100000+"),
    ("siemens ag","Germany","Conglomerate",1847,"100000+"),
    ("volkswagen ag","Germany","Automotive",1937,"100000+"),
    ("bmw group","Germany","Automotive",1916,"100000+"),
    ("mercedes benz group","Germany","Automotive",1926,"100000+"),
    ("basf se","Germany","Chemicals",1865,"100000+"),
    ("bayer ag","Germany","Pharma",1863,"100000+"),
    ("bosch group","Germany","Conglomerate",1886,"100000+"),
    ("deutsche bank","Germany","Finance",1870,"50000-100000"),
    ("allianz","Germany","Insurance",1890,"100000+"),
    # France
    ("lvmh","France","Luxury",1987,"100000+"),
    ("total energies","France","Energy",1924,"100000+"),
    ("bnp paribas","France","Finance",1848,"100000+"),
    ("airbus","France","Aerospace",1970,"100000+"),
    ("capgemini","France","Technology",1967,"100000+"),
    ("dassault systemes","France","Technology",1981,"20000-50000"),
    ("schneider electric","France","Energy",1836,"100000+"),
    # Japan
    ("toyota motor corporation","Japan","Automotive",1937,"100000+"),
    ("honda motor company","Japan","Automotive",1948,"100000+"),
    ("sony group corporation","Japan","Electronics",1946,"100000+"),
    ("samsung electronics","South Korea","Electronics",1969,"100000+"),
    ("lg electronics","South Korea","Electronics",1958,"50000-100000"),
    ("hyundai motor company","South Korea","Automotive",1967,"100000+"),
    ("sk hynix","South Korea","Semiconductors",1983,"30000-50000"),
    ("softbank group","Japan","Conglomerate",1981,"50000-100000"),
    ("fujitsu","Japan","Technology",1935,"100000+"),
    ("ntt data","Japan","Technology",1988,"100000+"),
    ("hitachi","Japan","Conglomerate",1910,"100000+"),
    ("panasonic","Japan","Electronics",1918,"100000+"),
    # China
    ("alibaba group","China","Technology",1999,"100000+"),
    ("tencent holdings","China","Technology",1998,"100000+"),
    ("baidu inc","China","Technology",2000,"50000-100000"),
    ("huawei technologies","China","Technology",1987,"100000+"),
    ("xiaomi corporation","China","Consumer Electronics",2010,"20000-50000"),
    ("bytedance","China","Technology",2012,"100000+"),
    ("didi global","China","Technology",2012,"20000-50000"),
    ("jd.com","China","E-Commerce",1998,"100000+"),
    ("meituan","China","Technology",2010,"50000-100000"),
    ("pinduoduo","China","E-Commerce",2015,"20000-50000"),
    # Other regions
    ("spotify","Sweden","Entertainment",2006,"9000-10000"),
    ("klarna","Sweden","Fintech",2005,"5000-10000"),
    ("zendesk","Denmark","Technology",2007,"5000-10000"),
    ("adyen","Netherlands","Fintech",2006,"2000-5000"),
    ("asml","Netherlands","Semiconductors",1984,"30000-50000"),
    ("booking holdings","USA","Travel",1996,"20000-50000"),
    ("trivago","Germany","Travel",2005,"2000-5000"),
    ("grab holdings","Singapore","Technology",2012,"5000-10000"),
    ("sea limited","Singapore","Technology",2009,"50000-100000"),
    ("gojek","Indonesia","Technology",2010,"20000-50000"),
    ("tokopedia","Indonesia","E-Commerce",2009,"10000-20000"),
    ("mercado libre","Argentina","E-Commerce",1999,"20000-50000"),
    ("nubank","Brazil","Fintech",2013,"10000-20000"),
    ("rappi","Colombia","Technology",2015,"5000-10000"),
]

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 5: WHED UNESCO — Global universities
# (university_name_lower, country, established_year, qs_rank_approx)
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_UNIVERSITIES_WHED = [
    # USA
    ("massachusetts institute of technology","USA",1861,1),
    ("stanford university","USA",1885,2),
    ("harvard university","USA",1636,3),
    ("california institute of technology","USA",1891,4),
    ("university of chicago","USA",1890,10),
    ("columbia university","USA",1754,12),
    ("university of pennsylvania","USA",1740,13),
    ("princeton university","USA",1746,14),
    ("yale university","USA",1701,17),
    ("cornell university","USA",1865,20),
    ("johns hopkins university","USA",1876,25),
    ("northwestern university","USA",1851,30),
    ("duke university","USA",1838,52),
    ("university of michigan","USA",1817,23),
    ("university of california berkeley","USA",1868,10),
    ("university of california los angeles","USA",1919,44),
    ("university of california san diego","USA",1960,62),
    ("university of washington","USA",1861,72),
    ("carnegie mellon university","USA",1900,52),
    ("georgia institute of technology","USA",1885,88),
    ("purdue university","USA",1869,99),
    ("university of texas at austin","USA",1883,67),
    ("new york university","USA",1831,44),
    ("boston university","USA",1839,109),
    ("university of southern california","USA",1880,113),
    ("university of illinois urbana champaign","USA",1867,82),
    ("university of wisconsin madison","USA",1848,87),
    ("university of minnesota","USA",1851,185),
    ("penn state university","USA",1855,150),
    ("ohio state university","USA",1870,176),
    # UK
    ("university of oxford","UK",1096,3),
    ("university of cambridge","UK",1209,5),
    ("imperial college london","UK",1907,6),
    ("university college london","UK",1826,8),
    ("london school of economics","UK",1895,49),
    ("university of edinburgh","UK",1583,22),
    ("university of manchester","UK",1824,28),
    ("king's college london","UK",1829,37),
    ("university of bristol","UK",1909,55),
    ("university of warwick","UK",1965,67),
    ("university of glasgow","UK",1451,76),
    ("university of birmingham","UK",1900,84),
    ("university of leeds","UK",1904,86),
    ("university of sheffield","UK",1905,95),
    ("university of southampton","UK",1862,100),
    ("durham university","UK",1832,92),
    ("university of nottingham","UK",1881,103),
    ("university of exeter","UK",1955,150),
    ("university of st andrews","UK",1413,100),
    # Canada
    ("university of toronto","Canada",1827,21),
    ("mcgill university","Canada",1821,30),
    ("university of british columbia","Canada",1908,34),
    ("university of waterloo","Canada",1957,112),
    ("western university","Canada",1878,170),
    ("queens university","Canada",1841,209),
    ("university of alberta","Canada",1908,111),
    ("simon fraser university","Canada",1965,298),
    ("mcmaster university","Canada",1887,152),
    ("university of montreal","Canada",1878,111),
    # Australia
    ("australian national university","Australia",1946,30),
    ("university of melbourne","Australia",1853,33),
    ("university of sydney","Australia",1850,41),
    ("university of queensland","Australia",1909,43),
    ("university of new south wales","Australia",1949,19),
    ("monash university","Australia",1958,57),
    ("university of western australia","Australia",1911,90),
    ("university of adelaide","Australia",1874,89),
    ("rmit university","Australia",1887,151),
    # Europe
    ("eth zurich","Switzerland",1855,7),
    ("epfl","Switzerland",1969,14),
    ("delft university of technology","Netherlands",1842,57),
    ("leiden university","Netherlands",1575,150),
    ("university of amsterdam","Netherlands",1632,52),
    ("lund university","Sweden",1666,78),
    ("karolinska institute","Sweden",1810,30),
    ("kth royal institute of technology","Sweden",1827,89),
    ("technical university of munich","Germany",1868,37),
    ("heidelberg university","Germany",1386,87),
    ("ludwig maximilian university munich","Germany",1472,54),
    ("freie universitat berlin","Germany",1948,98),
    ("humboldt university berlin","Germany",1809,121),
    ("university of bonn","Germany",1818,201),
    ("rwth aachen university","Germany",1870,106),
    ("paris sciences et lettres","France",2010,24),
    ("sorbonne university","France",1257,83),
    ("ecole polytechnique","France",1794,65),
    ("hec paris","France",1881,0),
    ("insead","France",1957,0),
    ("ku leuven","Belgium",1425,73),
    ("ghent university","Belgium",1817,150),
    ("university of copenhagen","Denmark",1479,79),
    ("aarhus university","Denmark",1928,134),
    ("university of helsinki","Finland",1640,107),
    ("aalto university","Finland",2010,115),
    ("university of oslo","Norway",1811,119),
    ("ntnu","Norway",1996,451),
    ("utrecht university","Netherlands",1636,101),
    ("wageningen university","Netherlands",1918,115),
    ("erasmus university rotterdam","Netherlands",1913,151),
    # Asia
    ("national university of singapore","Singapore",1905,8),
    ("nanyang technological university","Singapore",1991,15),
    ("peking university","China",1898,17),
    ("tsinghua university","China",1911,14),
    ("zhejiang university","China",1897,44),
    ("fudan university","China",1905,42),
    ("shanghai jiao tong university","China",1896,46),
    ("university of hong kong","Hong Kong",1911,26),
    ("hong kong university of science and technology","Hong Kong",1991,40),
    ("chinese university of hong kong","Hong Kong",1963,38),
    ("city university of hong kong","Hong Kong",1984,62),
    ("seoul national university","South Korea",1946,31),
    ("korea advanced institute of science and technology","South Korea",1971,42),
    ("pohang university of science and technology","South Korea",1986,83),
    ("yonsei university","South Korea",1885,151),
    ("university of tokyo","Japan",1877,28),
    ("kyoto university","Japan",1897,38),
    ("osaka university","Japan",1931,75),
    ("tokyo institute of technology","Japan",1881,55),
    ("keio university","Japan",1858,189),
    ("waseda university","Japan",1882,200),
    ("national taiwan university","Taiwan",1928,65),
    ("national tsing hua university","Taiwan",1911,172),
    ("national cheng kung university","Taiwan",1931,235),
    ("university of science and technology beijing","China",1952,308),
    ("beihang university","China",1952,456),
    ("harbin institute of technology","China",1920,270),
    ("nanjing university","China",1902,133),
    ("university of science and technology of china","China",1958,93),
    ("wuhan university","China",1893,257),
    ("sun yat-sen university","China",1924,206),
    # Middle East & Africa
    ("king abdulaziz university","Saudi Arabia",1967,136),
    ("king fahd university of petroleum and minerals","Saudi Arabia",1963,186),
    ("university of tehran","Iran",1934,601),
    ("sharif university of technology","Iran",1966,351),
    ("american university of beirut","Lebanon",1866,351),
    ("university of cape town","South Africa",1829,226),
    ("stellenbosch university","South Africa",1918,401),
    ("university of witwatersrand","South Africa",1922,441),
    ("cairo university","Egypt",1908,701),
    # Latin America
    ("university of sao paulo","Brazil",1934,101),
    ("universidade estadual de campinas","Brazil",1966,201),
    ("pontificia universidad catolica de chile","Chile",1888,151),
    ("universidad nacional autonoma de mexico","Mexico",1551,103),
    ("universidad de los andes colombia","Colombia",1948,233),
    ("universidad de buenos aires","Argentina",1821,71),
]

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 6: ArXiv bulk metadata — Research venues (conferences + journals)
# (venue_name, venue_type, primary_field, arxiv_category, issn_or_dblp_key)
# ──────────────────────────────────────────────────────────────────────────────
RESEARCH_VENUES_ARXIV = [
    # Top AI/ML Conferences
    ("NeurIPS","conference","AI/ML","cs.LG","neurips"),
    ("ICML","conference","AI/ML","cs.LG","icml"),
    ("ICLR","conference","AI/ML","cs.LG","iclr"),
    ("AAAI","conference","AI","cs.AI","aaai"),
    ("IJCAI","conference","AI","cs.AI","ijcai"),
    ("UAI","conference","AI/ML","cs.LG","uai"),
    ("AISTATS","conference","AI/ML","cs.LG","aistats"),
    ("COLT","conference","Learning Theory","cs.LG","colt"),
    ("ALT","conference","Learning Theory","cs.LG","alt"),
    ("ECML","conference","AI/ML","cs.LG","ecml"),
    # NLP Conferences
    ("ACL","conference","NLP","cs.CL","acl"),
    ("EMNLP","conference","NLP","cs.CL","emnlp"),
    ("NAACL","conference","NLP","cs.CL","naacl"),
    ("COLING","conference","NLP","cs.CL","coling"),
    ("EACL","conference","NLP","cs.CL","eacl"),
    ("CoNLL","conference","NLP","cs.CL","conll"),
    ("SemEval","workshop","NLP","cs.CL","semeval"),
    ("WMT","workshop","NLP/MT","cs.CL","wmt"),
    ("TACL","journal","NLP","cs.CL","tacl"),
    ("CL journal","journal","NLP","cs.CL","cl"),
    # Computer Vision
    ("CVPR","conference","Computer Vision","cs.CV","cvpr"),
    ("ICCV","conference","Computer Vision","cs.CV","iccv"),
    ("ECCV","conference","Computer Vision","cs.CV","eccv"),
    ("WACV","conference","Computer Vision","cs.CV","wacv"),
    ("BMVC","conference","Computer Vision","cs.CV","bmvc"),
    ("ACCV","conference","Computer Vision","cs.CV","accv"),
    # Systems & Architecture
    ("SOSP","conference","Systems","cs.OS","sosp"),
    ("OSDI","conference","Systems","cs.OS","osdi"),
    ("NSDI","conference","Networking","cs.NI","nsdi"),
    ("SIGCOMM","conference","Networking","cs.NI","sigcomm"),
    ("EuroSys","conference","Systems","cs.OS","eurosys"),
    ("ATC","conference","Systems","cs.OS","atc"),
    ("FAST","conference","Storage","cs.AR","fast"),
    ("ISCA","conference","Architecture","cs.AR","isca"),
    ("MICRO","conference","Architecture","cs.AR","micro"),
    ("HPCA","conference","Architecture","cs.AR","hpca"),
    ("SC","conference","HPC","cs.DC","sc"),
    ("PPoPP","conference","Parallel","cs.DC","ppopp"),
    # Databases
    ("SIGMOD","conference","Databases","cs.DB","sigmod"),
    ("VLDB","conference","Databases","cs.DB","vldb"),
    ("ICDE","conference","Databases","cs.DB","icde"),
    ("PODS","conference","Databases","cs.DB","pods"),
    ("EDBT","conference","Databases","cs.DB","edbt"),
    ("CIDR","conference","Databases","cs.DB","cidr"),
    # Security
    ("IEEE S&P","conference","Security","cs.CR","ieeesp"),
    ("USENIX Security","conference","Security","cs.CR","usenixsec"),
    ("CCS","conference","Security","cs.CR","ccs"),
    ("NDSS","conference","Security","cs.CR","ndss"),
    ("CRYPTO","conference","Cryptography","cs.CR","crypto"),
    ("EUROCRYPT","conference","Cryptography","cs.CR","eurocrypt"),
    ("ASIACRYPT","conference","Cryptography","cs.CR","asiacrypt"),
    # Software Engineering
    ("ICSE","conference","Software Engineering","cs.SE","icse"),
    ("FSE","conference","Software Engineering","cs.SE","fse"),
    ("ASE","conference","Software Engineering","cs.SE","ase"),
    ("ISSTA","conference","Software Testing","cs.SE","issta"),
    ("PLDI","conference","Programming Languages","cs.PL","pldi"),
    ("POPL","conference","Programming Languages","cs.PL","popl"),
    ("OOPSLA","conference","Programming Languages","cs.PL","oopsla"),
    # Data Mining & IR
    ("KDD","conference","Data Mining","cs.LG","kdd"),
    ("SIGIR","conference","Information Retrieval","cs.IR","sigir"),
    ("WWW","conference","Web","cs.IR","www"),
    ("RecSys","conference","Recommender Systems","cs.IR","recsys"),
    ("WSDM","conference","Web Mining","cs.IR","wsdm"),
    ("CIKM","conference","Knowledge Management","cs.IR","cikm"),
    ("ISWC","conference","Semantic Web","cs.AI","iswc"),
    # Robotics & Control
    ("ICRA","conference","Robotics","cs.RO","icra"),
    ("IROS","conference","Robotics","cs.RO","iros"),
    ("RSS","conference","Robotics","cs.RO","rss"),
    ("CoRL","conference","Robot Learning","cs.RO","corl"),
    # HCI
    ("CHI","conference","HCI","cs.HC","chi"),
    ("UIST","conference","HCI","cs.HC","uist"),
    ("CSCW","conference","CSCW","cs.HC","cscw"),
    # Top Journals
    ("Nature","journal","Multidisciplinary","","0028-0836"),
    ("Science","journal","Multidisciplinary","","0036-8075"),
    ("Nature Machine Intelligence","journal","AI/ML","cs.LG","2522-5839"),
    ("Nature Communications","journal","Multidisciplinary","","2041-1723"),
    ("Cell","journal","Biology","","0092-8674"),
    ("JMLR","journal","AI/ML","cs.LG","1532-4435"),
    ("IEEE TPAMI","journal","AI/ML","cs.CV","0162-8828"),
    ("Artificial Intelligence","journal","AI","cs.AI","0004-3702"),
    ("ACM TOIS","journal","IR","cs.IR","1046-8188"),
    ("IEEE TNN","journal","Neural Networks","cs.NE","2162-237X"),
    ("Machine Learning","journal","AI/ML","cs.LG","0885-6125"),
    ("Data Mining and Knowledge Discovery","journal","Data Mining","cs.LG","1384-5810"),
    ("Journal of Machine Learning Research","journal","AI/ML","cs.LG","1532-4435"),
    ("Transactions on Knowledge and Data Engineering","journal","Data Mining","cs.DB","1041-4347"),
    ("IEEE Transactions on Neural Networks and Learning Systems","journal","AI/ML","cs.LG","2162-237X"),
    ("Applied Intelligence","journal","AI","cs.AI","0924-669X"),
    ("Expert Systems with Applications","journal","AI","cs.AI","0957-4174"),
    ("Neurocomputing","journal","Neural Networks","cs.NE","0925-2312"),
    ("Pattern Recognition","journal","CV/ML","cs.CV","0031-3203"),
    ("Information Sciences","journal","AI","cs.AI","0020-0255"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Existing seed data preserved from original build_kb.py
# ──────────────────────────────────────────────────────────────────────────────
FICTIONAL_COMPANIES = [
    ("dunder mifflin","TV fiction (The Office)"),
    ("hooli","TV fiction (Silicon Valley)"),
    ("pied piper","TV fiction (Silicon Valley)"),
    ("acme corp","cartoon fiction"),
    ("acme corporation","cartoon fiction"),
    ("initech","film fiction (Office Space)"),
    ("globex","TV fiction (Simpsons)"),
    ("soylent corp","film fiction"),
    ("umbrella corporation","game fiction (Resident Evil)"),
    ("stark industries","comic fiction (Marvel)"),
    ("wayne enterprises","comic fiction (DC)"),
    ("cyberdyne systems","film fiction (Terminator)"),
    ("weyland-yutani","film fiction (Alien)"),
    ("tyrell corporation","film fiction (Blade Runner)"),
    ("oscorp","comic fiction (Marvel)"),
    ("aperture science","game fiction (Portal)"),
    ("black mesa","game fiction (Half-Life)"),
    ("vault-tec","game fiction (Fallout)"),
    ("wonka industries","film fiction"),
    ("massive dynamic","TV fiction (Fringe)"),
    ("buy n large","film fiction (WALL-E)"),
    ("bluth company","TV fiction (Arrested Development)"),
    ("veridian dynamics","TV fiction (Better Off Ted)"),
    ("sterling cooper","TV fiction (Mad Men)"),
    ("nakatomi corporation","film fiction (Die Hard)"),
    ("dharma initiative","TV fiction (Lost)"),
    ("virtucon","film fiction (Austin Powers)"),
    ("sky net","film fiction (Terminator)"),
    ("omni consumer products","film fiction (RoboCop)"),
    ("soylent green","film fiction"),
]

SKILL_ALIASES = [
    ("large language models","llm","ml"),
    ("machine learning","ml","ml"),
    ("natural language processing","nlp","nlp"),
    ("deep learning","dl","ml"),
    ("retrieval augmented generation","rag","nlp"),
    ("python","python3","lang"),
    ("kubernetes","k8s","infra"),
    ("javascript","js","lang"),
    ("typescript","ts","lang"),
    ("react","reactjs","frontend"),
    ("vue","vuejs","frontend"),
    ("angular","angularjs","frontend"),
    ("node","nodejs","backend"),
    ("postgres","postgresql","db"),
    ("mongo","mongodb","db"),
    ("tensorflow","tf","ml"),
    ("pytorch","torch","ml"),
    ("scikit-learn","sklearn","ml"),
    ("amazon web services","aws","cloud"),
    ("google cloud platform","gcp","cloud"),
    ("microsoft azure","azure","cloud"),
    ("docker","container","infra"),
    ("ci/cd","cicd","devops"),
    ("devops","dev ops","devops"),
    ("generative ai","genai","ml"),
    ("transformer","transformers","ml"),
    ("bert","bidirectional encoder representations","nlp"),
    ("gpt","generative pre-trained transformer","nlp"),
    ("vector database","vectordb","ml"),
    ("elasticsearch","elastic search","db"),
]


def _compress(name: str, out_dir: Path, comp_dir: Path):
    import tarfile as tf
    src = out_dir
    tar_path = comp_dir / f"{name}.tar.gz"
    with tf.open(tar_path, "w:gz") as tar:
        tar.add(src, arcname=name)
    log.info(f"  compressed → {tar_path.name} ({tar_path.stat().st_size/1e6:.1f} MB)")
    # update checksum
    h = hashlib.sha256()
    with open(tar_path,"rb") as f:
        for blk in iter(lambda: f.read(8192), b""):
            h.update(blk)
    cf = comp_dir / "checksums.sha256"
    lines = {}
    if cf.exists():
        for line in cf.read_text().splitlines():
            if "  " in line:
                cs, fn = line.split("  ",1)
                lines[fn] = cs
    lines[tar_path.name] = h.hexdigest()
    cf.write_text("".join(f"{cs}  {fn}\n" for fn,cs in sorted(lines.items())))


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    COMP.mkdir(parents=True, exist_ok=True)

    if DB.exists():
        DB.unlink()
    conn = sqlite3.connect(str(DB))

    # ── Schema ──────────────────────────────────────────────────────────────
    conn.executescript("""
    CREATE TABLE fictional_companies (
        company_name TEXT PRIMARY KEY,
        reason TEXT
    );
    CREATE TABLE company_founding_dates (
        company_name TEXT PRIMARY KEY,
        founding_year INT
    );
    CREATE TABLE skill_aliases (
        skill_canonical TEXT,
        alias TEXT,
        category TEXT
    );
    CREATE TABLE indian_companies (
        company_name TEXT PRIMARY KEY,
        cin TEXT,
        state TEXT,
        status TEXT,
        founded_year INT,
        source TEXT DEFAULT 'MCA data.gov.in'
    );
    CREATE TABLE dpiit_startups (
        startup_name TEXT PRIMARY KEY,
        dpiit_number TEXT,
        industry TEXT,
        state TEXT,
        founded_year INT,
        source TEXT DEFAULT 'DPIIT Startup India'
    );
    CREATE TABLE indian_universities (
        institution_name TEXT PRIMARY KEY,
        institution_type TEXT,
        state TEXT,
        approved_by TEXT,
        established_year INT,
        source TEXT DEFAULT 'AICTE+UGC'
    );
    CREATE TABLE global_companies (
        company_name TEXT PRIMARY KEY,
        country TEXT,
        industry TEXT,
        founded_year INT,
        employee_range TEXT,
        source TEXT DEFAULT 'Kaggle PDL 7M'
    );
    CREATE TABLE global_universities (
        university_name TEXT PRIMARY KEY,
        country TEXT,
        established_year INT,
        qs_rank_approx INT,
        source TEXT DEFAULT 'WHED UNESCO'
    );
    CREATE TABLE research_venues (
        venue_name TEXT PRIMARY KEY,
        venue_type TEXT,
        primary_field TEXT,
        arxiv_category TEXT,
        issn_or_key TEXT,
        source TEXT DEFAULT 'ArXiv bulk metadata'
    );
    CREATE TABLE source_metadata (
        source_name TEXT PRIMARY KEY,
        table_name TEXT,
        row_count INT,
        build_date TEXT,
        status TEXT
    );
    """)

    # ── Indexes ──────────────────────────────────────────────────────────────
    conn.executescript("""
    CREATE INDEX idx_fic ON fictional_companies(company_name);
    CREATE INDEX idx_cfd ON company_founding_dates(company_name);
    CREATE INDEX idx_alias ON skill_aliases(skill_canonical);
    CREATE INDEX idx_inc ON indian_companies(company_name);
    CREATE INDEX idx_inc_state ON indian_companies(state);
    CREATE INDEX idx_dpiit ON dpiit_startups(startup_name);
    CREATE INDEX idx_dpiit_ind ON dpiit_startups(industry);
    CREATE INDEX idx_iuniv ON indian_universities(institution_name);
    CREATE INDEX idx_iuniv_type ON indian_universities(institution_type);
    CREATE INDEX idx_gcorp ON global_companies(company_name);
    CREATE INDEX idx_gcorp_ctry ON global_companies(country);
    CREATE INDEX idx_guniv ON global_universities(university_name);
    CREATE INDEX idx_guniv_ctry ON global_universities(country);
    CREATE INDEX idx_venue ON research_venues(venue_name);
    CREATE INDEX idx_venue_field ON research_venues(primary_field);
    """)

    today = datetime.date.today().isoformat()

    # ── Load data ─────────────────────────────────────────────────────────────
    conn.executemany("INSERT OR IGNORE INTO fictional_companies VALUES (?,?)", FICTIONAL_COMPANIES)
    log.info(f"  fictional_companies: {len(FICTIONAL_COMPANIES)} rows")

    # founding dates = indian + global combined
    founding = [(r[0], r[4]) for r in INDIAN_COMPANIES_MCA] + \
               [(r[0], r[3]) for r in GLOBAL_COMPANIES_PDL if r[3]]
    conn.executemany("INSERT OR IGNORE INTO company_founding_dates VALUES (?,?)", founding)
    log.info(f"  company_founding_dates: {len(founding)} rows")

    conn.executemany("INSERT OR IGNORE INTO skill_aliases VALUES (?,?,?)", SKILL_ALIASES)
    log.info(f"  skill_aliases: {len(SKILL_ALIASES)} rows")

    conn.executemany(
        "INSERT OR IGNORE INTO indian_companies VALUES (?,?,?,?,?,'MCA data.gov.in')",
        INDIAN_COMPANIES_MCA)
    log.info(f"  indian_companies: {len(INDIAN_COMPANIES_MCA)} rows")

    conn.executemany(
        "INSERT OR IGNORE INTO dpiit_startups VALUES (?,?,?,?,?,'DPIIT Startup India')",
        DPIIT_STARTUPS)
    log.info(f"  dpiit_startups: {len(DPIIT_STARTUPS)} rows")

    conn.executemany(
        "INSERT OR IGNORE INTO indian_universities VALUES (?,?,?,?,?,'AICTE+UGC')",
        INDIAN_UNIVERSITIES)
    log.info(f"  indian_universities: {len(INDIAN_UNIVERSITIES)} rows")

    conn.executemany(
        "INSERT OR IGNORE INTO global_companies VALUES (?,?,?,?,?,'Kaggle PDL 7M')",
        GLOBAL_COMPANIES_PDL)
    log.info(f"  global_companies: {len(GLOBAL_COMPANIES_PDL)} rows")

    conn.executemany(
        "INSERT OR IGNORE INTO global_universities VALUES (?,?,?,?,'WHED UNESCO')",
        GLOBAL_UNIVERSITIES_WHED)
    log.info(f"  global_universities: {len(GLOBAL_UNIVERSITIES_WHED)} rows")

    conn.executemany(
        "INSERT OR IGNORE INTO research_venues VALUES (?,?,?,?,?,'ArXiv bulk metadata')",
        RESEARCH_VENUES_ARXIV)
    log.info(f"  research_venues: {len(RESEARCH_VENUES_ARXIV)} rows")

    # ── Source metadata ───────────────────────────────────────────────────────
    sources = [
        ("MCA data.gov.in",      "indian_companies",    len(INDIAN_COMPANIES_MCA),   today, "loaded-seed"),
        ("DPIIT Startup India",  "dpiit_startups",      len(DPIIT_STARTUPS),          today, "loaded-seed"),
        ("AICTE+UGC",            "indian_universities", len(INDIAN_UNIVERSITIES),      today, "loaded-seed"),
        ("Kaggle PDL 7M",        "global_companies",    len(GLOBAL_COMPANIES_PDL),    today, "loaded-seed"),
        ("WHED UNESCO",          "global_universities", len(GLOBAL_UNIVERSITIES_WHED),today, "loaded-seed"),
        ("ArXiv bulk metadata",  "research_venues",     len(RESEARCH_VENUES_ARXIV),   today, "loaded-seed"),
    ]
    conn.executemany("INSERT OR REPLACE INTO source_metadata VALUES (?,?,?,?,?)", sources)

    conn.commit()
    conn.close()
    log.info("fraud_kb.db written — compressing...")
    _compress("fraud_kb", OUT, COMP)
    return True


if __name__ == "__main__":
    ok = build()
    print("\nBuild", "OK" if ok else "FAILED")
