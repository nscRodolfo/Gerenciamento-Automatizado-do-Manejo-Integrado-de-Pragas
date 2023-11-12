package my.aplication.manejointeligentedepragas;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;

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

import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;

import com.example.manejointeligentedepragas.R;

import org.json.JSONException;
import org.json.JSONObject;

import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Cadastro extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    private DrawerLayout drawerLayout;

    public Button btnCadastrar;

    public EditText tbNome;
    public EditText tbTelefone;
    public EditText tbEmail;
    public EditText tbConEmail;
    public EditText tbSenha;
    public EditText tbConSenha;
    public RadioGroup rgtipoUsu;

    public ProgressBar pbc;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cadastro);

        setTitle("MIP² | Cadastro");

        pbc = findViewById(R.id.progressBarC);

        tbNome = findViewById(R.id.tbNome);
        tbTelefone = findViewById(R.id.tbTelefone);
        tbEmail = findViewById(R.id.tbEmailC);
        tbConEmail = findViewById(R.id.tbConEmail);
        tbSenha = findViewById(R.id.tbSenhaC);
        tbConSenha = findViewById(R.id.tbConSenha);

        rgtipoUsu = findViewById(R.id.rgTipousuario);

        btnCadastrar = findViewById(R.id.btnCadastrar);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_cadastro);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_cadastro);
        NavigationView navigationView = findViewById(R.id.nav_view_cadastro);
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

        btnCadastrar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    Toast.makeText(Cadastro.this,"Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
                }else { // se tem acesso à internet chama cadastrar

                    cadastrar();
                }
            }
        });
    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Toast.makeText(Cadastro.this,"Para acessar seu perfil, faça login!", Toast.LENGTH_LONG).show();
                break;
            case R.id.drawerProp:
                Toast.makeText(Cadastro.this,"Para acessar as propriedades, faça login!", Toast.LENGTH_LONG).show();
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

    public void cadastrar()
    {
        final Intent k = new Intent(Cadastro.this, Entrar.class);

        String nome = tbNome.getText().toString();
        String telefone = tbTelefone.getText().toString();
        String email = tbEmail.getText().toString();
        String conEmail = tbConEmail.getText().toString();
        String senha = tbSenha.getText().toString();
        String conSenha = tbConSenha.getText().toString();
        String tipoUsuario = null;

        switch (rgtipoUsu.getCheckedRadioButtonId())
        {
            case R.id.rbFuncioario:
                tipoUsuario = "Funcionario";
                break;
            case R.id.rbProprietario:
                tipoUsuario = "Produtor";
                break;
        }
        if (tipoUsuario == null) {
            Toast.makeText(Cadastro.this,"É necessário escolher um tipo de usuário!", Toast.LENGTH_LONG).show();
        }else if (nome.isEmpty()){
            tbNome.setError("Nome é obrigatório!");
        }else if (telefone.isEmpty()){
            tbTelefone.setError("Telefone é obrigatório!");
        }else if (telefone.length() < 10){
            tbTelefone.setError("Telefone informado não é válido");
        }else if (email.isEmpty()){
            tbEmail.setError("E-mail é obrigatório!");
        }else if (!isEmailValid(email)){
            tbEmail.setError("E-mail não é valido!");
        }else if (conEmail.isEmpty()){
            tbConEmail.setError("Confirmar o e-mail é obrigatório!");
        }else if (!(email.equals(conEmail))){
            tbConEmail.setError("Os e-mails informados não são iguais!");
        }else if (senha.isEmpty()) {
            tbSenha.setError("Senha é obrigatório");
        }else if (senha.length() < 6){
            tbSenha.setError("A senha precisa ter mais que 6 digitos");
        }else if (conSenha.isEmpty()){
            tbConSenha.setError("Confirmar a senha é obrigatório!");
        }else if (!(senha.equals(conSenha))){
            tbConSenha.setError("As senhas informadas não são iguais!");
        }else{
            pbc.setVisibility(View.VISIBLE);
            senha = convertPassMd5(senha);
            String url = "https://mip.software/phpapp/cadastrar.php?Nome="+nome+
                    "&&Telefone="+telefone+"&&Email="+email+"&&Senha="+senha+"&&tipoUsu="+tipoUsuario;
            RequestQueue queue = Volley.newRequestQueue(this);

            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {
                @Override
                public void onResponse(String response) {

                    // Parsing json
                    try {
                        JSONObject Obj = new JSONObject(response);
                        boolean confirmacao = Obj.getBoolean("confirmacao");

                        if(confirmacao){
                            pbc.setVisibility(View.GONE);
                            Toast.makeText(Cadastro.this, "Cadastro realizado com Sscesso!",Toast.LENGTH_LONG).show();
                            startActivity(k);
                        }else{
                            pbc.setVisibility(View.GONE);
                            Toast.makeText(Cadastro.this, "E-mail já cadastrado!",Toast.LENGTH_LONG).show();
                        }

                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(Cadastro.this,error.toString(), Toast.LENGTH_LONG).show();
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


}
