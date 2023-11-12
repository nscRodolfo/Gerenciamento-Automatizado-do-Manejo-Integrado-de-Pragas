package my.aplication.manejointeligentedepragas;

import android.app.Dialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.drawable.ColorDrawable;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import java.security.MessageDigest;
import java.math.BigInteger;
import java.security.NoSuchAlgorithmException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.model.UsuarioModel;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class Entrar extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    private DrawerLayout drawerLayout;

    public Button btnLogin;

    public EditText edt_login;
    public EditText edt_senha;
    TextView esqueciSenha;


    Integer Cod_Propriedade;


    private Dialog mDialog;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_entrar);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_entrar);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_entrar);
        NavigationView navigationView = findViewById(R.id.nav_view_entrar);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();

        View headerView = navigationView.getHeaderView(0);

        ImageView imageView = headerView.findViewById(R.id.imageMenu);
        //Drawable drawable = getResources().getDrawable(R.mipmap.ic_launcher);
        //imageView.setImageDrawable(drawable);

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        //nomeMenu.setVisibility(View.GONE);
        nomeMenu.setText("Monitoramento Inteligente de Pragas");

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setVisibility(View.GONE);


        setTitle("MIP² | Login");

        edt_login = findViewById(R.id.tbEmail2);
        edt_senha = findViewById(R.id.tbSenha);
        esqueciSenha = findViewById(R.id.tvEsqueci);

        btnLogin = findViewById(R.id.btnEntrarL);
        btnLogin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    Toast.makeText(Entrar.this,"Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
                }else { // se tem acesso à internet chama logar
                    openDialog();
                    logar();
                }
            }
        });

        esqueciSenha.setClickable(true);
        esqueciSenha.setFocusable(true);

        esqueciSenha.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent k = new Intent(Entrar.this, EsqueciSenha.class);
                startActivity(k);
            }
        });


    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            Intent i = new Intent(Entrar.this,MainActivity.class);
            startActivity(i);
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Toast.makeText(Entrar.this,"Para acessar seu perfil, faça login!", Toast.LENGTH_LONG).show();
                break;
            case R.id.drawerProp:
                Toast.makeText(Entrar.this,"Para acessar as propriedades, faça login!", Toast.LENGTH_LONG).show();
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;
            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }

    private void logar()
    {
        final String login = edt_login.getText().toString().trim(); // trim- não passar espaço em branco
        String aux = edt_senha.getText().toString().trim();
        final String senha = convertPassMd5(aux);
        //Toast.makeText(Entrar.this,"MD5 da senha: "+senha, Toast.LENGTH_LONG).show();

        if (login.isEmpty()){
            edt_login.setError("E-mail é obrigatório!");
            mDialog.dismiss();
        }else if (!isEmailValid(login)){
            edt_login.setError("E-mail não é valido!");
            mDialog.dismiss();
        }else if (senha.isEmpty()){
            edt_senha.setError("Senha é obrigatório");
           mDialog.dismiss();
        }else {
            //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();

            //endereçodeipv4  dinamico - pode alterar quando mudar de rede
            //String url = "https://192.168.0.20/AppMIP/Entrar.php?email="+login;

            String url = "https://mip.software/phpapp/Entrar.php?email="+login;


            RequestQueue queue = Volley.newRequestQueue(this);
            final String finalSenha = senha;
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        JSONObject obj = array.getJSONObject(0);
                        UsuarioModel u = new UsuarioModel();
                        u.setCod_Usuario(obj.getInt("Cod_Usuario"));
                        u.setEmail(obj.getString("Email"));
                        u.setSenha(obj.getString("Senha"));
                        u.setNome(obj.getString("Nome"));
                        u.setTelefone(obj.getString("Telefone"));
                        u.setTipo(obj.getString("Tipo"));


                        if(u.getSenha().trim().equals(finalSenha)){
                            boolean c = new Controller_Usuario(getBaseContext()).addUsuario(u);
                            // adiciona no banco local o Usuario
                            Toast.makeText(Entrar.this, "Logado com sucesso!",Toast.LENGTH_LONG).show();
                            mDialog.dismiss();
                            final Intent k = new Intent(Entrar.this, Propriedades.class);
                            startActivity(k);
                        }else{
                            Toast.makeText(Entrar.this, "Dados invalidos!",Toast.LENGTH_LONG).show();
                            mDialog.dismiss();
                        }
                    } catch (JSONException e) {
                        Toast.makeText(Entrar.this, "Email não cadastrado",Toast.LENGTH_LONG).show();
                        mDialog.dismiss();
                        logar();
                    }

                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(Entrar.this,error.toString(), Toast.LENGTH_LONG).show();
                   mDialog.dismiss();
                }
            }));
        }
    }



    public static boolean isEmailValid(String email) {
        String expression = "^[\\w\\.-]+@([\\w\\-]+\\.)+[A-Z]{2,4}$";
        Pattern pattern = Pattern.compile(expression, Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(email);
        return matcher.matches();
    }

    // converte String para hash MD5
    public static String convertPassMd5(String pass) {
        String password = null;
        MessageDigest mdEnc;
        try {
            mdEnc = MessageDigest.getInstance("MD5");
            mdEnc.update(pass.getBytes(), 0, pass.length());
            pass = new BigInteger(1, mdEnc.digest()).toString(16);
            while (pass.length() < 32) {
                pass = "0" + pass;
            }
            password = pass;
        } catch (NoSuchAlgorithmException e1) {
            e1.printStackTrace();
        }
        return password;
    }

    public void openDialog(){
        mDialog = new Dialog(this);
        //vamos remover o titulo da Dialog
        mDialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        //vamos carregar o xml personalizado
        mDialog.setContentView(R.layout.dialog);
        //DEixamos transparente
        mDialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        // não permitimos fechar esta dialog
        mDialog.setCancelable(false);
        //temos a instancia do ProgressBar!
        final ProgressBar progressBar = ProgressBar.class.cast(mDialog.findViewById(R.id.progressBar));

        mDialog.show();

        // mDialog.dismiss(); -> para fechar a dialog

    }


}
