CREATE TABLE users (
	id SERIAL PRIMARY KEY, 
	email VARCHAR(255) UNIQUE NOT NULL,
	password_hash TEXT NOT NULL,
	is_active BOOLEAN DEFAULT TRUE,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	uptaded_at TIMESTAMP
);

CREATE TABLE roles(
	id SERIAL PRIMARY KEY,
	name VARCHAR(50) UNIQUE NOT NULL,
	description TEXT,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)

CREATE TABLE permissions(
	id SERIAL PRIMARY KEY,
	name VARCHAR(100) UNIQUE NOT NULL,
	Description TEXT
)

CREATE TABLE user_roles(
	id SERIAL PRIMARY KEY,
	user_id INT REFERENCES users(id) ON DELETE CASCADE,
	role_id INT REFERENCES roles(id) ON DELETE cascade,
	assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE role_permissions (
	id SERIAL PRIMARY KEY,
	role_id INT REFERENCES roles(id) ON DELETE CASCADE,
	permission_id INT REFERENCES permission(id) ON DELETE CASCADE

)

CREATE TABLE refresh_tokens (
	id SERIAL Primary KEY,
	user_id INT REFERENCES users(id),
	token_hash TEXT not null,
	expires_at timestamp,
	revoked_at timestamp,
	created_at timestamp default current_timestamp
);

CREATE TABLE audit_logs (
	id SERIAL PRIMARY KEY,
	user_id INT REFERENCES users(id),
	action VARCHAR(255),
	entity varchar(100),
	entity_id int,
	old_value text,
	new_Value text,
	ip_address varchar(100),
	created_At timestamp default current_timestamp

);

CREATE TABLE settings (
	id SERIAL PRIMARY KEY,
	key varchar(100) unique,
	value text,
	description text,
	uptaded_at timestamp
);

CREATE TABLE notification(
	id serial Primary key,
	user_id int references users(id),
	type varchar(50),
	tittle varchar(255),
	message text,
	is_read boolean default FALSE,
	created_at timestamp DEFAULT current_timestamp
);

CREATE TABLE password_resets(
	id serial primary key,
	user_id int references users(id),
	token text,
	expires_At timestamp
)


CREATE TABLE categories (
	id serial primary key,
	name varchar(255),
	parent_category_id INT REFERENCES categories(id)
);

CREATE TABLE products(
	id serial primary key,
	name text,
	description text,
	base_price numeric,
	category_id int references categories(id),
	created_at timestamp default current_timestamp

);

Create table products_variants(
	id serial primary key,
	product_id int references products(id),
	sku varchar(100),
	color varchar(50),
	size varchar(50)
);

create table inventory(
	id serial primary key,
	variant_id int references product_variants(id),
	stock_quantity int,
	warehouse_location varchar(255)
);

create table orders (
	id serial primary key,
	user_id INT references users(id),
	order_date timestamp default current_timestamp,
	total_amount numeric,
	status varchar(50)
);

create table order_items (
	id serial primary key,
	orders_int references orders(id),
	variant_id int references product_variants(id),
	quantity INT,
	price NUMERIC
);

Create table reviews (
	id serial primary key,
	user_id int references users(id)
	product_id int references products(id),
	rating INT,
	tittle text,
	content text,
	created_At timestamp default current_timestamp
);

Create table wishlist (
	id serial primary key,
	user_id int references users(id),
	product_id int references products(id)
)