# StockVision-AI-Sales-Forecasting-Demand-Prediction

1. Permbledhja- Analiza
StockVision AI - Eshte nje ekosistem i shkalles siperore (enterprise-grade) i projektuar për të kryer analize te sakte te sentimentit mbi te dhenat e shitjeve te Amazon. Duke ndare logjiken e biznesit nga infrastruktura permes nje arkitekture te shtresezuar, sistemi garanton modularitet, testueshmeri dhe siguri te nivelit te larte. 
Ai integron heuristikat e avancuara të mësimit automatik me një skeme te normalizuar te bazes se te dhenave qe perfshin 24 entitete.
---------------------------------------------------------------------------------------------------------------------------
2. Paradigma ArkitekturoreSistemi ndjek parimet e Clean Architecture, duke imponuar rrjedha strikte të varesive:Shtresa e Kontrolloreve (Controllers): Nderfaqja me shtresen e transportit, duke zbatuar dizajnin "contract-first" per API.
3. Shtresa e Shërbimeve (Services): Ekzekuton logjikën komplekse të domenit dhe orkestron "pipeline-in" e inferences se modelit të mesimit automatik.Shtresa e Depove (Repositories): Implementon modelin DAO (Data Access Object) per te abstraktuar mekanizmat e persistences, duke siguruar pavarësinë nga baza e të dhënave dhe mbrojtje kunder SQL Injection.
-------------------------------------------------------------------------------------------------------------------------
3. Topologjia e Skemes Relacionale Shtresa e persistencës se te dhenave eshte normalizuar ne formën e tretë normale (3NF), e strukturuar në 24 entitete:Infrastruktura Bazë (10 entitete): Menaxhimi i identitetit, kontrolli i aksesit me baze rolesh (RBAC), log-et e pandryshueshme te auditimit dhe menaxhimi i sesioneve.Skema specifike e domenit (14 entitete): Menaxhimi i ciklit të jetes se produkteve, rrjedhat e shitjeve dhe metriket e analizes se sentimentit.
------------------------------------------------------------------------------------------------------------------------
4. Siguria dhe Pajtueshmëria
Siguria nuk eshte nje detaj anesor, por një shtylle qendrore arkitekturore:Autentifikimi dhe Autorizimi: Implementim i plote i JWT me rotacion të "Access" dhe "Refresh" token-ave.Integriteti Kriptografik: Hashimi i fjalëkalimeve permes algoritmit Argon2 ose Bcrypt me kripezim (salting) të pershpejtuar nga hardueri.Validimi i Inputit: Siguri strikte e tipit (type-safety) perrmes skemave Pydantic, duke neutralizuar vektorët e injektimit.Menaxhimi i Sekreteve: Politik e rrepte kunder hard-kodimit te kredencialeve; perdorimi i dotenv për injektimin e variablave specifike te mjedisit.
-------------------------------------------------------------------------------------------------------------------------
5. Pipeline i Inferences se MLSistemit perdor nje model te trajnuar te hapesires vektoriale per te kryer klasifikimin e sentimentit. Shtresa e shërbimeve ekspozon një "endpoint" të fuqishem per inference, duke mbështetur perpunimin e te dhenave në kohe reale.AlgoritmiSaktësia (Precision)RecallF1-ScoreLogistic Regression0.940.920.93Linear SVC0.950.940.9456. Stack-u TeknologjikRuntime: Python 3.12+ (Asynchronous I/O)Framework për API: FastAPIPersistenca: SQL 16ORM: SQLAlchemy 2.0 (me Alembic Migrations)Motorri i ML: Scikit-Learn7. Protokolli i ZhvillimitKonfigurimi i Mjedisit: Inicializoni variablat e mjedisit duke perdorur shabllonin .env.example.Kontejnerizimi: docker-compose up --buildMigrimi i Bazes: Ekzekutoni alembic upgrade head për të materializuar skemen me 24 tabela.Verifikimi: Vizitoni /docs për dokumentacionin e gjeneruar automatikisht te OpenAPI.
-----------------------------------------------------------------------------------------------------------------------
6. Ky kuader i përmbahet standardeve të percaktuara në Udhëzimet e Arkitektures dhe Inxhinierise se Softuerit 2025, duke siguruar pajtueshmeri me praktikat industriale dhe ekselencën akademike.



                      © 2026 Ekipi i Inxhinierise se STOCKVISION-AI. Te gjitha te drejtat te rezervuara.
